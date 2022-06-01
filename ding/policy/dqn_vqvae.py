from typing import List, Dict, Any, Tuple
from collections import namedtuple
import copy
import torch

from ding.torch_utils import Adam, to_device, to_tensor
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, get_nstep_return_data, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn
from ding.model.template.action_vqvae import ActionVQVAE

from ding.utils import RunningMeanStd
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from ding.envs.common import affine_transform

@POLICY_REGISTRY.register('dqn-vqvae')
class DQNVQVAEPolicy(Policy):
    r"""
    Overview:
        Policy class of DQN-VQVAE algorithm.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      dqn            | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     False          | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     False          | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        7  ``nstep``            int      1,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  | ``learn.update``   int      3              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        9  | ``learn.multi``    bool     False          | whether to use multi gpu during
           | ``_gpu``
        10 | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.target_``  int      100            | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        13 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        14 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        15 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        16 | ``other.eps.type`` str      exp            | exploration rate decay type            | Support ['exp',
                                                                                                 | 'linear'].
        17 | ``other.eps.``     float    0.95           | start value of exploration rate        | [0,1]
           | ``start``
        18 | ``other.eps.``     float    0.1            | end value of exploration rate          | [0,1]
           | ``end``
        19 | ``other.eps.``     int      10000          | decay length of exploration            | greater than 0. set
           | ``decay``                                                                           | decay=10000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        == ==================== ======== ============== ======================================== =======================
    """

    config = dict(
        type='dqn-vqvae',
        cuda=False,
        on_policy=False,
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        priority_vqvae=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight_vqvae=False,
        discount_factor=0.97,
        nstep=1,
        original_action_shape=2,
        # (str) Action space type
        action_space='hybrid',  # ['continuous', 'hybrid']
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            rl_batch_size=512,
            vqvae_batch_size=512,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # Frequency of target network update.
            target_update_freq=500,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=8,
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
            replay_buffer=dict(replay_buffer_size=10000, ),
        ),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``, initialize the optimizer, algorithm arguments, main \
            and target model.
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

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        # self._target_model = model_wrap(
        #     self._target_model,
        #     wrapper_name='target',
        #     update_type='momentum',
        #     update_kwargs={'theta': self._cfg.learn.target_update_theta}
        # )

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
        self._running_mean_std_predict_loss = RunningMeanStd(epsilon=1e-4)
        self.c_percentage_bound_lower = -1 * torch.ones([6])
        self.c_percentage_bound_upper = torch.ones([6])

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overview:
            Forward computation graph of learn mode(updating policy).
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``action``, ``reward``, ``next_obs``, ``done``
            - optional: ``value_gamma``, ``IS``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``total_loss``, ``priority``
            - optional: ``action_distribution``
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
            result = self._vqvae_model.train(data)
 
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

            # print(loss_dict['reconstruction_loss'])
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
                if self._cfg.priority_type_vqvae=='reward':
                    # TODO: data['reward'] is nstep reward, take the true reward 
                    # first max-min normalization, transform to [0, 1] priority should be non negtive (>=0)
                    # then scale to [0.2, 1], to make sure all data can be used to update vqvae
                    reward_normalization = (data['reward'][0] - data['reward'][0].min())/(data['reward'][0].max() - data['reward'][0].min()+1e-8)
                    reward_normalization =  (1-self._cfg.priority_vqvae_min)* reward_normalization + self._cfg.priority_vqvae_min

                return {
                    'priority': reward_normalization.tolist(), 
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
                if self._cfg.recompute_latent_action:
                    if self._cfg.rl_reconst_loss_weight:
                        result = self._vqvae_model.train(data)
                        reconstruction_loss_none_reduction = result['recons_loss_none_reduction']
                        data['latent_action'] = result['quantized_index'].clone().detach()
                    else:
                        quantized_index = self._vqvae_model.encode({'action': data['action']})
                        data['latent_action'] = quantized_index.clone().detach()
                        # print(torch.unique(data['latent_action']))

                if self._cuda:
                    data = to_device(data, self._device)
                # ====================
                # Q-learning forward
                # ====================
                self._learn_model.train()
                self._target_model.train()
                # Current q value (main model)
                q_value = self._learn_model.forward(data['obs'])['logit']
                # Target q value
                with torch.no_grad():
                    target_q_value = self._target_model.forward(data['next_obs'])['logit']
                    # print(torch.unique(data['next_obs'][:,0]))
                    # print(torch.unique(target_q_value[:,0]))
                    # Max q value action (main model)
                    if self._cfg.learn.constrain_action is True:
                        # TODO(pu)
                        quantized_index = self.visualize_latent(save_histogram=False)  # NOTE: visualize_latent
                        constrain_action = torch.unique(torch.from_numpy(quantized_index))
                        next_q_value = self._learn_model.forward(data['next_obs'])['logit']
                        target_q_action = torch.argmax(next_q_value[:, constrain_action], dim=-1)
                    else:
                        target_q_action = self._learn_model.forward(data['next_obs'])['action']
                        # print(torch.unique(target_q_action))

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

                loss, td_error_per_sample = q_nstep_td_error(
                    data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma
                )

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
                loss_dict['critic_loss'] = loss.item()

                q_value_dict = {}
                q_value_dict['q_value'] = q_value.mean().item()

                return {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'td_error': td_error_per_sample.abs().mean(),
                    **loss_dict,
                    **q_value_dict,
                    'total_grad_norm_rl': total_grad_norm_rl,
                    # 'rewrad_run': data['rewrad_run'].mean().item(),
                    # 'rewrad_ctrl': data['rewrad_ctrl'].mean().item(),

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
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``, initialize algorithm arguments and collect_model, \
            enable the eps_greedy_sample for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()
        self._warm_up_stop = False

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        """
        Overview:
            Forward computation graph of collect mode(collect training data), with eps_greedy for exploration.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting policy_output(action) for the interaction with \
                env and the constructing of transition.
        ArgumentsKeys:
            - necessary: ``obs``
        ReturnsKeys
            - necessary: ``logit``, ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        # for subprocess case
        data = data.float()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
            # here output['action'] is the out of DQN, is discrete action
            output['latent_action'] = copy.deepcopy(output['action'])
            if self._cuda:
                output = to_device(output, self._device)

            # backup
            # TODO(pu): decode into original hybrid actions, here data is obs
            # this is very important to generate self.obs_encoding using in decode phase
            # output['action'] = self._vqvae_model.decode_with_obs(output['action'], data})['recons_action']

            if self._cfg.action_space == 'hybrid':
                # backup
                # recons_action = self._vqvae_model.decode_without_obs(output['action'])
                # output['action'] = {
                #     'action_type': recons_action['recons_action']['disc'],
                #     'action_args': recons_action['recons_action']['cont']
                # }
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
                # output['action']  = self._vqvae_model.decode_with_obs(output['action'])
                output['action'] = self._vqvae_model.decode(output['action'])['recons_action']
                
                # debug
                # latents = to_device(torch.arange(64), 'cuda')
                # recons_action = self._vqvae_model.decode(latents)['recons_action']
                # print(recons_action.max(0), recons_action.min(0),recons_action.mean(0), recons_action.std(0))

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
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, policy_output: Dict[str, Any], timestep: namedtuple) -> Dict[str, Any]:
        """
        Overview:
            Generate a transition(e.g.: <s, a, s', r, d>) for this algorithm training.
        Arguments:
            - obs (:obj:`Any`): Env observation.
            - policy_output (:obj:`Dict[str, Any]`): The output of policy collect mode(``self._forward_collect``),\
                including at least ``action``.
            - timestep (:obj:`namedtuple`): The output after env step(execute policy output action), including at \
                least ``obs``, ``reward``, ``done``, (here obs indicates obs after env step).
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
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ArgumentsKeys:
            - necessary: ``obs``
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
            output['latent_action'] = copy.deepcopy(output['action'])

            # TODO(pu): decode into original hybrid actions, here data is obs
            # this is very important to generate self.obs_encoding using in decode phase
            # output['action'] = self._vqvae_model.decode_with_obs(output['action'], data})['recons_action']

            if self._cfg.action_space == 'hybrid':
                # recons_action = self._vqvae_model.decode_without_obs(output['action'])
                # output['action'] = {
                #     'action_type': recons_action['recons_action']['disc'],
                #     'action_args': recons_action['recons_action']['cont']
                # }
                recons_action = self._vqvae_model.decode(output['action'])
                output['action'] = {
                    'action_type': recons_action['recons_action']['action_type'],
                    'action_args': recons_action['recons_action']['action_args']
                }
            else:
                # output['action'] = self._vqvae_model.decode_without_obs(output['action'])['recons_action']
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

    def visualize_latent(self, save_histogram=True, name=0, granularity=0.1):
        if self.cfg.action_space == 'continuous':
            # continuous action, now only for hopper env: 3 dim cont
            xx, yy, zz = np.meshgrid(
                np.arange(-1, 1, granularity), np.arange(-1, 1, granularity), np.arange(-1, 1, granularity)
            )
            cnt = int((2 / granularity)) ** 3
            action_samples = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).reshape(cnt, 3)
        elif self.cfg.action_space == 'hybrid':  
            # hybrid action, now only for gym_hybrid env: 3 dim discrete, 2 dim cont
            xx, yy = np.meshgrid(np.arange(-1, 1, granularity), np.arange(-1, 1, granularity))
            cnt = int((2 / granularity)) ** 2
            action_samples = np.array([xx.ravel(), yy.ravel()]).reshape(cnt, 2)

            action_samples_type1 = np.concatenate(
                [np.tile(np.array([1, 0, 0]), cnt).reshape(cnt, 3), action_samples], axis=-1
            )
            action_samples_type2 = np.concatenate(
                [np.tile(np.array([0, 1, 0]), cnt).reshape(cnt, 3), action_samples], axis=-1
            )
            action_samples_type3 = np.concatenate(
                [np.tile(np.array([0, 0, 1]), cnt).reshape(cnt, 3), action_samples], axis=-1
            )
            action_samples = np.concatenate([action_samples_type1, action_samples_type2, action_samples_type3])

        # encoding = self._vqvae_model.encoder(torch.Tensor(action_samples).to(torch.device('cuda')))
        # quantized_index, quantized_inputs, vq_loss, _, _ = self._vqvae_model.vq_layer(encoding)

        with torch.no_grad():
            # action_embedding = self._get_action_embedding(data)
            encoding = self._vqvae_model.encoder(to_device(torch.Tensor(action_samples), self._device))
            quantized_index = self._vqvae_model.vq_layer.encode(encoding)

        if save_histogram:
            fig = plt.figure()

            # Fixing bin edges
            HIST_BINS = np.linspace(0, self._cfg.model.action_shape - 1, self._cfg.model.action_shape)

            # the histogram of the data
            n, bins, patches = plt.hist(
                quantized_index.detach().cpu().numpy(), HIST_BINS, density=False, facecolor='g', alpha=0.75
            )

            plt.xlabel('Latent Discrete action')
            plt.ylabel('Count')
            plt.title('Histogram of Latent Discrete action')
            plt.grid(True)
            plt.show()

            if isinstance(name, int):
                plt.savefig(f'latent_histogram_iter{name}.png')
                print(f'save latent_histogram_iter{name}.png')
            elif isinstance(name, str):
                plt.savefig('latent_histogram_' + name + '.png')
                print('latent_histogram_' + name + '.png')
        else:
            return quantized_index.detach().cpu().numpy()

    def visualize_embedding_table(self, save_dis_map=True, name=0):
        embedding_table = self._vqvae_model.vq_layer.embedding.weight.detach().cpu()
        """cos distance"""
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        dis = [[] for i in range(embedding_table.shape[0])]
        for i in range(embedding_table.shape[0]):
            for j in list(range(embedding_table.shape[0])):
                dis[i].append(cos(embedding_table[i], embedding_table[j]).numpy())
        if save_dis_map:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('Embedding table CosineSimilarity')
            plt.imshow(dis)
            plt.colorbar()
            plt.show()
            if isinstance(name, int):
                plt.savefig(f'embedding_table_CosineSimilarity_iter{name}.png')
                print(f'save embedding_table_CosineSimilarity_iter{name}.png')
            elif isinstance(name, str):
                plt.savefig('embedding_table_CosineSimilarity_' + name + '.png')
                print('save embedding_table_CosineSimilarity_' + name + '.png')
        else:
            return np.array(dis)

# debug
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Hopper-v3 dqn episode0_latent_actions')
# plt.plot(episode0_latent_actions)
# plt.show()
# plt.savefig(f'hopper-v3_dqn_episode0_latent_actions.png')
