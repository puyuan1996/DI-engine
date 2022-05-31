from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import copy
import torch
from torch.distributions import Categorical
import logging
from easydict import EasyDict
from ding.torch_utils import Adam, to_device
from ding.utils.data import default_collate, default_decollate
from ding.rl_utils import q_nstep_td_data, q_nstep_sql_td_error, get_nstep_return_data, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_utils import default_preprocess_learn
from ding.model.template.action_vqvae import ActionVQVAE


@POLICY_REGISTRY.register('sql-vqvae-episode')
class SQLVQVAEEPISODEPolicy(Policy):
    r"""
    Overview:
        Policy class of SQL algorithm.
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='sql-vqvae-episode',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.97,
        # (int) N-step reward for target q_value estimation
        nstep=1,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,  # after the batch data come into the learner, train with the data for 3 times
            batch_size=64,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
            alpha=0.1,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=8,  # collect 8 samples and put them in collector
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                # (int) Decay length(env step)
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, )
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._priority_vqvae = self._cfg.priority_vqvae
        self._priority_IS_weight_vqvae = self._cfg.priority_IS_weight_vqvae
        # Optimizer
        # NOTE:
        if self._cfg.learn.rl_clip_grad is True:
            self._optimizer = Adam(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                grad_clip_type=self._cfg.learn.grad_clip_type,
                clip_value=self._cfg.learn.grad_clip_value
            )
        else:
            self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._alpha = self._cfg.learn.alpha
        # use wrapper instead of plugin
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0  # count iterations
        self._vqvae_model = ActionVQVAE(
            self._cfg.original_action_shape,
            self._cfg.model.action_shape,  #K
            self._cfg.vqvae_embedding_dim,  #D
            self._cfg.vqvae_hidden_dim,
            self._cfg.vq_loss_weight,
            is_ema=self._cfg.is_ema,
            is_ema_target=self._cfg.is_ema_target,
            eps_greedy_nearest=self._cfg.eps_greedy_nearest,
            cont_reconst_l1_loss=self._cfg.cont_reconst_l1_loss,
            cont_reconst_smooth_l1_loss=self._cfg.cont_reconst_smooth_l1_loss,
            categorical_head_for_cont_action=self._cfg.categorical_head_for_cont_action,
            n_atom=self._cfg.n_atom,
            gaussian_head_for_cont_action=self._cfg.gaussian_head_for_cont_action,
            embedding_table_onehot=self._cfg.embedding_table_onehot,
            vqvae_return_weight=self._cfg.vqvae_return_weight,
        )
        self._vqvae_model = to_device(self._vqvae_model, self._device)
        # NOTE:
        if self._cfg.learn.vqvae_clip_grad is True:
                self._optimizer_vqvae = Adam(
                self._vqvae_model.parameters(),
                lr=self._cfg.learn.learning_rate_vae,
                grad_clip_type=self._cfg.learn.grad_clip_type,
                clip_value=self._cfg.learn.grad_clip_value
        )
        else:
            self._optimizer_vqvae = Adam(
                self._vqvae_model.parameters(),
                lr=self._cfg.learn.learning_rate_vae,
            )

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        ### warmup phase: train VAE ###
        if 'warm_up' in data[0].keys() and data[0]['warm_up'] is True:
            loss_dict = {}
            data = default_preprocess_learn(
                data,
                use_priority=self._cfg.priority,
                use_priority_IS_weight=self._cfg.priority_IS_weight,
                ignore_done=self._cfg.learn.ignore_done,
                use_nstep=False
            )
            if self._cuda:
                data = to_device(data, self._device)

            # ====================
            # train vae
            # ====================
            result = self._vqvae_model.train(data, warmup=True)
 
            if self._cfg.gaussian_head_for_cont_action:
                # debug
                sigma = result['sigma']
                # print(sigma.max(0), sigma.min(0), sigma.mean(0), sigma.std(0))
                loss_dict['sigma.max'] = sigma.max(0)
                loss_dict['sigma.min'] = sigma.min(0)
                loss_dict['sigma.mean'] = sigma.mean(0)
                loss_dict['sigma.std'] = sigma.std(0)

            loss_dict['total_vqvae_loss'] = result['total_vqvae_loss'].item()
            loss_dict['reconstruction_loss'] = result['recons_loss'].item()
            loss_dict['vq_loss'] = result['vq_loss'].item()
            loss_dict['embedding_loss'] = result['embedding_loss'].item()
            loss_dict['commitment_loss'] = result['commitment_loss'].item()
            
            #  print(loss_dict['reconstruction_loss'])
            if loss_dict['reconstruction_loss'] < self._cfg.learn.reconst_loss_stop_value:
                self._warm_up_stop = True
                
            # vae update
            self._optimizer_vqvae.zero_grad()
            result['total_vqvae_loss'].backward()
            self._optimizer_vqvae.step()

            # NOTE:visualize_latent, now it's only for env hopper and gym_hybrid
            # quantized_index = self.visualize_latent(save_histogram=False)
            # cos_similarity = self.visualize_embedding_table(save_dis_map=False)

            return {
                'cur_lr': self._optimizer.defaults['lr'],
                # 'td_error': td_error_per_sample,
                **loss_dict,
                # **q_value_dict,
                # '[histogram]latent_action': quantized_index,
                # '[histogram]cos_similarity': cos_similarity,
            }
        ### VQVAE+RL phase ###
        else:
            self._forward_learn_cnt += 1
            loss_dict = {}
            q_value_dict = {}
            data = default_preprocess_learn(
                data,
                use_priority=self._priority,
                use_priority_IS_weight=self._cfg.priority_IS_weight,
                ignore_done=self._cfg.learn.ignore_done,
                use_nstep=True
            )
            # ====================
            # train VQVAE
            # ====================
            if data['vae_phase'][0].item() is True:
                if self._cuda:
                    data = to_device(data, self._device)

                if self._cfg.priority_type_vqvae=='reward':
                    # TODO: data['reward'] is nstep reward, take the true reward 
                    # first max-min normalization, transform to [0, 1] priority should be non negtive (>=0)
                    # then scale to [0.2, 1], to make sure all data can be used to update vqvae
                    reward_normalization = (data['reward'][0] - data['reward'][0].min())/(data['reward'][0].max() - data['reward'][0].min()+1e-8)
                    reward_normalization =  (1-self._cfg.priority_vqvae_min)* reward_normalization + self._cfg.priority_vqvae_min
                elif self._cfg.priority_type_vqvae=='return':
                    # TODO: data['return'] is the cumulative undiscounted reward
                    # first max-min normalization, transform to [0, 1] priority should be non negtive (>=0)
                    # then scale to [0.2, 1], to make sure all data can be used to update vqvae
                    return_normalization = (data['return'] - data['return'].min())/(data['return'].max() - data['return'].min()+1e-8)
                    return_normalization =  (1-self._cfg.priority_vqvae_min)* return_normalization + self._cfg.priority_vqvae_min
                if self._cfg.vqvae_return_weight:
                    # TODO: data['return'] is the cumulative undiscounted reward
                    # first max-min normalization, transform to [0, 1] priority should be non negtive (>=0)
                    # then scale to [0.2, 1], to make sure all data can be used to update vqvae
                    return_normalization = (data['return'] - data['return'].min())/(data['return'].max() - data['return'].min()+1e-8)
                    return_normalization =  (1-self._cfg.priority_vqvae_min)* return_normalization + self._cfg.priority_vqvae_min
                    data['return_normalization'] = return_normalization

                result = self._vqvae_model.train(data)

                loss_dict['total_vqvae_loss'] = result['total_vqvae_loss'].item()
                loss_dict['reconstruction_loss'] = result['recons_loss'].item()
                loss_dict['vq_loss'] = result['vq_loss'].item()
                loss_dict['embedding_loss'] = result['embedding_loss'].item()
                loss_dict['commitment_loss'] = result['commitment_loss'].item()

                # vae update
                self._optimizer_vqvae.zero_grad()
                result['total_vqvae_loss'].backward()
                total_grad_norm_vqvae = self._optimizer_vqvae.get_grad()
                self._optimizer_vqvae.step()

                
                # NOTE:visualize_latent, now it's only for env hopper and gym_hybrid
                # quantized_index = self.visualize_latent(save_histogram=False)
                # cos_similarity = self.visualize_embedding_table(save_dis_map=False)

                return {
                    'cur_lr': self._optimizer.defaults['lr'],
                    # 'td_error': td_error_per_sample,
                    **loss_dict,
                    # **q_value_dict,
                    'total_grad_norm_vqvae': total_grad_norm_vqvae,
                    # '[histogram]latent_action': quantized_index,
                    # '[histogram]cos_similarity': cos_similarity,
                }
            # ====================
            # train RL
            # ====================
            else:
                # ====================
                # relabel latent action
                # ====================
                if self._cuda:
                    data = to_device(data, self._device)

                # Representation shift correction (RSC)
                # update all latent action
                # result = self._vqvae_model.inference_without_obs({'action': data['action']})
                # data['latent_action'] = result['quantized_index'].clone().detach()

                if self._cfg.recompute_latent_action:
                    if self._cfg.rl_reconst_loss_weight:
                        result = self._vqvae_model.train(data)
                        reconstruction_loss_none_reduction = result['recons_loss_none_reduction']
                        data['latent_action'] = result['quantized_index'].clone().detach()
                    else:
                        quantized_index = self._vqvae_model.encode({'action': data['action']})
                        data['latent_action'] = quantized_index.clone().detach()
                        # print(torch.unique(data['latent_action']))

                # ====================
                # Q-learning forward
                # ====================
                self._learn_model.train()
                self._target_model.train()
                # Current q value (main model)
                q_value = self._learn_model.forward(data['obs'])['logit']
                with torch.no_grad():
                    # Target q value
                    target_q_value = self._target_model.forward(data['next_obs'])['logit']
                    # Max q value action (main model)
                    target_q_action = self._learn_model.forward(data['next_obs'])['action']
                
                # TODO: weight RL loss according to the reconstruct loss, because in 
                # In the area with large reconstruction loss, the action reconstruction is inaccurate, that is, the (\hat{x}, r) does not match, 
                # and the corresponding Q value is inaccurate. The update should be reduced to avoid wrong gradient.
                if self._cfg.rl_reconst_loss_weight:
                    # TODO:
                    # fist max-min normalization, transform to [0, 1]
                    reconstruction_loss_none_reduction_normalization = (reconstruction_loss_none_reduction - reconstruction_loss_none_reduction.min())/(reconstruction_loss_none_reduction.max() - reconstruction_loss_none_reduction.min()+1e-8)
                    # then scale to [0.2, 1], to make sure all data can be used to calculate gradients
                    # reconstruction_loss_weight = (1-self._cfg.rl_reconst_loss_weight_min) * reconstruction_loss_none_reduction_normalization + self._cfg.rl_reconst_loss_weight_min
                    # then scale to [1, 0.2], to make sure all data can be used to calculate gradients
                    reconstruction_loss_weight = -(1-self._cfg.rl_reconst_loss_weight_min) * reconstruction_loss_none_reduction_normalization + 1
                    data['weight'] = reconstruction_loss_weight
                
                # NOTE: RL learn policy in latent action space, so here using data['latent_action']
                data_n = q_nstep_td_data(
                    q_value, target_q_value, data['latent_action'].squeeze(-1), target_q_action, data['reward'],
                    data['done'], data['weight']
                )
                value_gamma = data.get('value_gamma')
                loss, td_error_per_sample, record_target_v = q_nstep_sql_td_error(
                    data_n, self._gamma, self._cfg.learn.alpha, nstep=self._nstep, value_gamma=value_gamma
                )
                record_target_v = record_target_v.mean()
                # ====================
                # Q-learning update
                # ====================
                self._optimizer.zero_grad()
                loss.backward()
                total_grad_norm_rl = self._optimizer.get_grad()
                if self._cfg.learn.multi_gpu:
                    self.sync_gradients(self._learn_model)
                self._optimizer.step()

                # =============
                # after update
                # =============
                self._target_model.update(self._learn_model.state_dict())

                q_value_dict = {}
                q_value_dict['q_value'] = q_value.mean().item()

                return {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': loss.item(),
                    'priority': td_error_per_sample.abs().tolist(),
                    'record_value_function': record_target_v,
                    # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
                    # '[histogram]action_distribution': data['action'],
                    **q_value_dict,
                    'total_grad_norm_rl': total_grad_norm_rl,
                }


    def _monitor_vars_learn(self) -> List[str]:
        ret = [
            'priority',
            'cur_lr',
            'critic_loss',
            'q_value',
            'td_error',
            'total_vqvae_loss',
            'reconstruction_loss',
            'embedding_loss',
            'commitment_loss',
            'vq_loss',
            'total_grad_norm_rl',
            'total_grad_norm_vqvae',
            'record_value_function'
            # 'predict_loss',
            # '[histogram]latent_action',
            # '[histogram]cos_similarity',
        ]
        return ret

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'vqvae_model': self._vqvae_model.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._vqvae_model.load_state_dict(state_dict['vqvae_model'])


    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._vqvae_model.load_state_dict(state_dict['vqvae_model'])


    def _load_state_dict_collect(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._vqvae_model.load_state_dict(state_dict['vqvae_model'])

    def _load_state_dict_eval(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._vqvae_model.load_state_dict(state_dict['vqvae_model'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Enable the eps_greedy_sample
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_multinomial_sample')
        self._collect_model.reset()
        self._warm_up_stop = False


    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        # for subprocess case
        data = data.float()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps, alpha=self._cfg.learn.alpha)
            
            # here output['action'] is the out of DQN, is discrete action
            output['latent_action'] = copy.deepcopy(output['action'])
            if self._cuda:
                output = to_device(output, self._device)

            if self._cfg.action_space == 'hybrid':
                recons_action = self._vqvae_model.decode(output['action'])
                output['action'] = {
                    'action_type': recons_action['recons_action']['action_type'],
                    'action_args': recons_action['recons_action']['action_args']
                }

                # NOTE: add noise in the original actions
                if self._cfg.learn.noise:
                    from ding.rl_utils.exploration import GaussianNoise
                    action = output['action']['action_args']
                    gaussian_noise = GaussianNoise(mu=0.0, sigma=self._cfg.learn.noise_sigma)
                    noise = gaussian_noise(
                        output['action']['action_args'].shape, output['action']['action_args'].device
                    )
                    if self._cfg.learn.noise_range is not None:
                        noise = noise.clamp(self._cfg.learn.noise_range['min'], self._cfg.learn.noise_range['max'])
                    action += noise
                    self.action_range = {'min': -1, 'max': 1}
                    if self.action_range is not None:
                        action = action.clamp(self.action_range['min'], self.action_range['max'])
                    output['action']['action_args'] = action
            else:
                output['action'] = self._vqvae_model.decode(output['action'])['recons_action']

                # NOTE: add noise in the original actions
                if self._cfg.learn.noise:
                    from ding.rl_utils.exploration import GaussianNoise
                    action = output['action']
                    gaussian_noise = GaussianNoise(mu=0.0, sigma=self._cfg.learn.noise_sigma)
                    noise = gaussian_noise(output['action'].shape, output['action'].device)
                    if self._cfg.learn.noise_range is not None:
                        noise = noise.clamp(self._cfg.learn.noise_range['min'], self._cfg.learn.noise_range['max'])
                    action += noise
                    self.action_range = {'min': -1, 'max': 1}
                    if self.action_range is not None:
                        action = action.clamp(self.action_range['min'], self.action_range['max'])
                    output['action'] = action
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. A train sample can be a processed transition(DQN with nstep TD) \
            or some continuous transitions(DRQN).
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): The list of training samples.

        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procecure by overriding this two methods and collector \
            itself.
        """
        for item in data:
            item['return'] = torch.stack([data[i]['reward'] for i in range(len(data))]).sum(0)

        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, policy_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - policy_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        if 'latent_action' in policy_output.keys():
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': policy_output['action'],
                'latent_action': policy_output['latent_action'],
                'reward': timestep.reward,
                # 'rewrad_run': timestep.info['rewrad_run'],
                # 'rewrad_ctrl': timestep.info['rewrad_ctrl'],
                # 'info': timestep.info,
                'done': timestep.done,
            }
        else:  # if random collect at fist
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': policy_output['action'],
                'latent_action': False,
                'reward': timestep.reward,
                'done': timestep.done,
            }
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
            # here output['action'] is the out of DQN, is discrete action
            # output['latent_action'] = output['action']  # TODO(pu)
            output['latent_action'] = copy.deepcopy(output['action'])

            if self._cfg.action_space == 'hybrid':
                recons_action = self._vqvae_model.decode(output['action'])
                output['action'] = {
                    'action_type': recons_action['recons_action']['action_type'],
                    'action_args': recons_action['recons_action']['action_args']
                }
            else:
                output['action'] = self._vqvae_model.decode(output['action'])['recons_action']
        
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For DQN, ``ding.model.template.q_learning.DQN``
        """
        return 'dqn', ['ding.model.template.q_learning']
