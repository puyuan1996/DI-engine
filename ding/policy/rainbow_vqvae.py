from typing import List, Dict, Any, Tuple, Union
import torch
import copy

from ding.torch_utils import Adam, to_device
from ding.rl_utils import dist_nstep_td_data, dist_nstep_td_error, get_train_sample, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn
from ding.model.template.action_vqvae import ActionVQVAE



@POLICY_REGISTRY.register('rainbow-vqvae')
class RainbowDQNVQVAEPolicy(DQNPolicy):
    r"""
    Overview:
        Rainbow DQN contain several improvements upon DQN, including:
            - target network
            - dueling architecture
            - prioritized experience replay
            - n_step return
            - noise net
            - distribution net

        Therefore, the RainbowDQNPolicy class inherit upon DQNPolicy class

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      rainbow        | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     True           | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  ``model.v_min``      float    -10            | Value of the smallest atom
                                                        | in the support set.
        6  ``model.v_max``      float    10             | Value of the largest atom
                                                        | in the support set.
        7  ``model.n_atom``     int      51             | Number of atoms in the support set
                                                        | of the value distribution.
        8  | ``other.eps``      float    0.05           | Start value for epsilon decay. It's
           | ``.start``                                 | small because rainbow use noisy net.
        9  | ``other.eps``      float    0.05           | End value for epsilon decay.
           | ``.end``
        10 | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        11 ``nstep``            int      3,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        12 | ``learn.update``   int      3              | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        == ==================== ======== ============== ======================================== =======================

    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='rainbow',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=True,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=True,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # random_collect_size=2000,
        model=dict(
            # (float) Value of the smallest atom in the support set.
            # Default to -10.0.
            v_min=-10,
            # (float) Value of the smallest atom in the support set.
            # Default to 10.0.
            v_max=10,
            # (int) Number of atoms in the support set of the
            # value distribution. Default to 51.
            n_atom=51,
        ),
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # (int) N-step reward for target q_value estimation
        nstep=3,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            batch_size=32,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=32,
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
                # (float) End value for epsilon decay, in [0, 1]. It's equals to `end` because rainbow uses noisy net.
                start=0.05,
                # (float) End value for epsilon decay, in [0, 1].
                end=0.05,
                # (int) Env steps of epsilon decay.
                decay=100000,
            ),
            replay_buffer=dict(
                # (int) Max size of replay buffer.
                replay_buffer_size=100000,
                # (float) Prioritization exponent.
                alpha=0.6,
                # (float) Importance sample soft coefficient.
                # 0 means no correction, while 1 means full correction
                beta=0.4,
                # (int) Anneal step for beta: 0 means no annealing. Defaults to 0
                anneal_step=100000,
            )
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Init the learner model of RainbowDQNPolicy

        Arguments:
            - learning_rate (:obj:`float`): the learning rate fo the optimizer
            - gamma (:obj:`float`): the discount factor
            - nstep (:obj:`int`): the num of n step return
            - v_min (:obj:`float`): value distribution minimum value
            - v_max (:obj:`float`): value distribution maximum value
            - n_atom (:obj:`int`): the number of atom sample point
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
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
        self._v_max = self._cfg.model.v_max
        self._v_min = self._cfg.model.v_min
        self._n_atom = self._cfg.model.n_atom

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
        # self._vqvae_model = VQVAE(2, 64, 64) #   action_dim: int, embedding_dim: int, num_embeddings: int,
        self._vqvae_model = ActionVQVAE(
            self._cfg.original_action_shape,
            self._cfg.model.action_shape,  #K
            self._cfg.vqvae_embedding_dim,  #D
            self._cfg.vqvae_hidden_dim,
            self._cfg.vq_loss_weight,
            is_ema=self._cfg.is_ema,
            is_ema_target=self._cfg.is_ema_target,
            eps_greedy_nearest=self._cfg.eps_greedy_nearest,
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
        """
        Overview:
            Forward and backward function of learn mode, acquire the data and calculate the loss and\
            optimize learner model

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'next_obs', 'reward', 'action']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): current learning rate
                - total_loss (:obj:`float`): the calculated loss
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
            # result = self._vqvae_model.train_without_obs(data)
            result = self._vqvae_model.train(data)

            loss_dict['total_vqvae_loss'] = result['total_vqvae_loss'].item()
            loss_dict['reconstruction_loss'] = result['recons_loss'].item()
            loss_dict['vq_loss'] = result['vq_loss'].item()
            loss_dict['embedding_loss'] = result['embedding_loss'].item()
            loss_dict['commitment_loss'] = result['commitment_loss'].item()

            # vae update
            self._optimizer_vqvae.zero_grad()
            result['total_vqvae_loss'].backward()
            self._optimizer_vqvae.step()
            # loss_dict['critic_loss'] = torch.Tensor([0]).item()
            # loss_dict['total_loss'] = torch.Tensor([0]).item()
            # q_value_dict = {}
            # q_value_dict['q_value'] = torch.Tensor([0]).item()
            # td_error_per_sample = torch.Tensor([0]).item()

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

                # result = self._vqvae_model.train_without_obs(data)
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

                # q_value_dict = {}
                # q_value_dict['q_value'] = torch.Tensor([0]).item()
                # td_error_per_sample = torch.Tensor([0]).item()
                
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

                quantized_index = self._vqvae_model.encode({'action': data['action']})
                data['latent_action'] = quantized_index.clone().detach()
                # print(torch.unique(data['latent_action']))

                if self._cuda:
                    data = to_device(data, self._device)

                # ====================
                # Rainbow forward
                # ====================
                self._learn_model.train()
                self._target_model.train()
                # reset noise of noisenet for both main model and target model
                self._reset_noise(self._learn_model)
                self._reset_noise(self._target_model)
                q_dist = self._learn_model.forward(data['obs'])['distribution']
                with torch.no_grad():
                    target_q_dist = self._target_model.forward(data['next_obs'])['distribution']
                    self._reset_noise(self._learn_model)
                    target_q_action = self._learn_model.forward(data['next_obs'])['action']
                value_gamma = data.get('value_gamma', None)

                # data = dist_nstep_td_data(
                #     q_dist, target_q_dist, data['action'], target_q_action, data['reward'], data['done'], data['weight']
                # )
                # NOTE: RL learn policy in latent action space, so here using data['latent_action']
                data = dist_nstep_td_data(
                            q_dist, target_q_dist, data['latent_action'].squeeze(-1), target_q_action, data['reward'], data['done'], data['weight']
                        )

                loss, td_error_per_sample = dist_nstep_td_error(
                    data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep, value_gamma=value_gamma
                )
                # ====================
                # Rainbow update
                # ====================
                self._optimizer.zero_grad()
                loss.backward()
                total_grad_norm_rl = self._optimizer.get_grad()
                self._optimizer.step()
                # =============
                # after update
                # =============
                self._target_model.update(self._learn_model.state_dict())
                # return {
                #     'cur_lr': self._optimizer.defaults['lr'],
                #     'total_loss': loss.item(),
                #     'priority': td_error_per_sample.abs().tolist(),
                # }
                loss_dict['critic_loss'] = loss.item()

                q_value_dict = {}
                q_value_dict['q_dist'] = q_dist.mean().item()

                return {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'td_error': td_error_per_sample.abs().mean(),
                    **loss_dict,
                    **q_value_dict,
                    'total_grad_norm_rl': total_grad_norm_rl,
                }

    def _monitor_vars_learn(self) -> List[str]:
        ret = [
            'cur_lr',
            'critic_loss',
            'q_dist',
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

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init moethod. Called by ``self.__init__``.
            Init traj and unroll length, collect model.

            .. note::
                the rainbow dqn enable the eps_greedy_sample, but might not need to use it, \
                    as the noise_net contain noise that can help exploration
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._nstep = self._cfg.nstep
        self._gamma = self._cfg.discount_factor
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Reset the noise from noise net and collect output according to eps_greedy plugin

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
        self._reset_noise(self._collect_model)
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
            # here output['action'] is the out of DQN, is discrete action
            output['latent_action'] = copy.deepcopy(output['action'])
            if self._cuda:
                output = to_device(output, self._device)

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
            # output['latent_action'] = output['action']  # TODO(pu)
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

    def _get_train_sample(self, traj: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data

        Arguments:
            - traj (:obj:`list`): The trajactory's buffer list

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = get_nstep_return_data(traj, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'rainbowdqn', ['ding.model.template.q_learning']

    def _reset_noise(self, model: torch.nn.Module):
        r"""
        Overview:
            Reset the noise of model

        Arguments:
            - model (:obj:`torch.nn.Module`): the model to reset, must contain reset_noise method
        """
        for m in model.modules():
            if hasattr(m, 'reset_noise'):
                m.reset_noise()
