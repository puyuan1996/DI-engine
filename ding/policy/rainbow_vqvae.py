from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
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

from ding.utils import RunningMeanStd
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from ding.envs.common import affine_transform


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
        type='rainbow-vqvae',
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

    def default_model(self) -> Tuple[str, List[str]]:
        return 'rainbowdqn', ['ding.model.template.q_learning']

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
        self._priority_vqvae = self._cfg.priority_vqvae
        self._priority_IS_weight_vqvae = self._cfg.priority_IS_weight_vqvae
        
        # Optimizer
        if self._cfg.learn.rl_clip_grad is True:
            if self._cfg.learn.rl_weight_decay is not None:
                self._optimizer = Adam(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                weight_decay=self._cfg.learn.rl_weight_decay,
                optim_type='adamw',
                grad_clip_type=self._cfg.learn.grad_clip_type,
                clip_value=self._cfg.learn.grad_clip_value
            )
            else:
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
        self._vqvae_model = ActionVQVAE(
            self._cfg.original_action_shape,
            self._cfg.latent_action_shape,  # K
            self._cfg.vqvae_embedding_dim,  # D
            self._cfg.vqvae_hidden_dim,
            beta=self._cfg.beta,
            vq_loss_weight=self._cfg.vq_loss_weight,  # TODO
            is_ema=self._cfg.is_ema,
            is_ema_target=self._cfg.is_ema_target,
            eps_greedy_nearest=self._cfg.eps_greedy_nearest,
            cont_reconst_l1_loss=self._cfg.cont_reconst_l1_loss,
            cont_reconst_smooth_l1_loss=self._cfg.cont_reconst_smooth_l1_loss,
            categorical_head_for_cont_action=self._cfg.categorical_head_for_cont_action,
            threshold_categorical_head_for_cont_action=self._cfg.threshold_categorical_head_for_cont_action,
            categorical_head_for_cont_action_threshold=self._cfg.categorical_head_for_cont_action_threshold,
            n_atom=self._cfg.n_atom,
            gaussian_head_for_cont_action=self._cfg.gaussian_head_for_cont_action,
            embedding_table_onehot=self._cfg.embedding_table_onehot,
            obs_regularization=self._cfg.obs_regularization,
            obs_shape=self._cfg.model.obs_shape,
            predict_loss_weight=self._cfg.predict_loss_weight,
            mask_pretanh=self._cfg.mask_pretanh,
            recons_loss_cont_weight = self._cfg.recons_loss_cont_weight
        )
        self._vqvae_model = to_device(self._vqvae_model, self._device)
        if self._cfg.learn.vqvae_clip_grad is True:
            if self._cfg.learn.rl_weight_decay is not None:
                self._optimizer_vqvae = Adam(
                    self._vqvae_model.parameters(),
                    lr=self._cfg.learn.learning_rate_vae,
                    weight_decay=self._cfg.learn.vqvae_weight_decay,
                    optim_type='adamw',
                    grad_clip_type=self._cfg.learn.grad_clip_type,
                    clip_value=self._cfg.learn.grad_clip_value
                )
            else:
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
            if self._cfg.obs_regularization:
                data['true_residual'] = data['next_obs'] - data['obs']
                result = self._vqvae_model.train_with_obs(data)
            else:
                result = self._vqvae_model.train(data)

            loss_dict['total_vqvae_loss'] = result['total_vqvae_loss'].item()
            loss_dict['reconstruction_loss'] = result['recons_loss'].item()
            loss_dict['vq_loss'] = result['vq_loss'].item()
            loss_dict['embedding_loss'] = result['embedding_loss'].item()
            loss_dict['commitment_loss'] = result['commitment_loss'].item()
            loss_dict['recons_action_probs_left_mask_proportion'] = float(result['recons_action_probs_left_mask_proportion'])
            loss_dict['recons_action_probs_right_mask_proportion'] = float(result['recons_action_probs_right_mask_proportion'])

            if self._cfg.obs_regularization:
                loss_dict['predict_loss'] = result['predict_loss'].item()

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

                if self._cfg.obs_regularization:
                    data['true_residual'] = data['next_obs'] - data['obs']
                    result = self._vqvae_model.train_with_obs(data)
                else:
                    result = self._vqvae_model.train(data)

                loss_dict['total_vqvae_loss'] = result['total_vqvae_loss'].item()
                loss_dict['reconstruction_loss'] = result['recons_loss'].item()
                loss_dict['vq_loss'] = result['vq_loss'].item()
                loss_dict['embedding_loss'] = result['embedding_loss'].item()
                loss_dict['commitment_loss'] = result['commitment_loss'].item()
                loss_dict['recons_action_probs_left_mask_proportion'] = float(result['recons_action_probs_left_mask_proportion'])
                loss_dict['recons_action_probs_right_mask_proportion'] = float(result['recons_action_probs_right_mask_proportion'])
                if self._cfg.obs_regularization:
                    loss_dict['predict_loss'] = result['predict_loss'].item()

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
                if self._cfg.recompute_latent_action:
                    if self._cfg.rl_reconst_loss_weight:
                        result = self._vqvae_model.train(data)
                        reconstruction_loss_none_reduction = result['recons_loss_none_reduction']
                        data['latent_action'] = result['quantized_index'].clone().detach()
                    else:
                        quantized_index = self._vqvae_model.encode(data)
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
            # '[histogram]latent_action',
            # '[histogram]cos_similarity',
        ]
        if self._cfg.obs_regularization:
            ret.append('predict_loss')
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
        self._warm_up_stop = False

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
            # here output['action'] is the out of Rainbow DQN, is discrete action
            output['latent_action'] = copy.deepcopy(output['action'])
            if self._cuda:
                output = to_device(output, self._device)

            if self._cfg.action_space == 'hybrid':
                # TODO(pu): decode into original hybrid actions, here data is obs
                recons_action = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data, 'threshold_phase': 'collect' in self._cfg.threshold_phase})
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
                output_action = torch.zeros([output['action'].shape[0], self._cfg.original_action_shape])
                if self._cuda:
                    output_action = to_device(output_action, self._device)
                # the latent of extreme_action, e.g. [64, 64+2**3) [64, 72)
                mask = output['action'].ge(self._cfg.latent_action_shape) & output['action'].le(self._cfg.model.action_shape)  # TODO

                # the usual latent of vqvae learned action
                output_action[~mask] = self._vqvae_model.decode({'quantized_index': output['action'].masked_select(~mask), 'obs': data.masked_select(~mask.unsqueeze(-1)).view(-1,self._cfg.model.obs_shape), 'threshold_phase': 'collect' in self._cfg.threshold_phase})[
                    'recons_action']

                if mask.sum() > 0:
                    # the latent of extreme_action, e.g. [64, 64+2**3) [64, 72) -> [0, 8)
                    extreme_action_index = output['action'].masked_select(mask) - self._cfg.latent_action_shape
                    from itertools import product
                    # NOTE:
                    # disc_to_cont: transform discrete action index to original continuous action
                    self.m = self._cfg.original_action_shape
                    self.n = 2
                    self.K =  self.n ** self.m
                    self.disc_to_cont = list(product(*[list(range(self.n)) for dim in range(self.m)] ))
                    # NOTE: disc_to_cont: transform discrete action index to original continuous action
                    extreme_action = torch.tensor([[-1. if k==0 else 1.for k in  self.disc_to_cont[int(one_extreme_action_index)]] for one_extreme_action_index in extreme_action_index])
                    if self._cuda:
                        extreme_action = to_device(extreme_action, self._device)
                    output_action[mask] = extreme_action

                output['action'] = output_action

                # debug
                # latents = to_device(torch.arange(64), 'cuda')
                # recons_action = self._vqvae_model.decode(latents)['recons_action']
                # print(recons_action.max(0), recons_action.min(0),recons_action.mean(0), recons_action.std(0))

                # NOTE: add noise in the original actions
                if self._cfg.learn.noise:
                    if self._cfg.learn.noise_augment_extreme_action:
                        if np.random.random() < self._cfg.learn.noise_augment_extreme_action_prob:
                            halved_prob = torch.ones_like(output['action'])/2
                            output['action'] = 2*torch.bernoulli(halved_prob)-1  # {-1,1}
                        else:
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
                    else:
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
        # for subprocess case
        data = data.float()
        with torch.no_grad():
            output = self._eval_model.forward(data)
            # here output['action'] is the out of Rainbow DQN, is discrete action
            # output['latent_action'] = output['action']  # TODO(pu)
            output['latent_action'] = copy.deepcopy(output['action'])

            if self._cfg.action_space == 'hybrid':
                recons_action = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data, 'threshold_phase': 'eval' in self._cfg.threshold_phase})
                output['action'] = {
                    'action_type': recons_action['recons_action']['action_type'],
                    'action_args': recons_action['recons_action']['action_args']
                }
            else:
                if not self._cfg.augment_extreme_action:
                    output['action'] = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data, 'threshold_phase': 'eval' in self._cfg.threshold_phase})[
                        'recons_action']
                else:
                    output_action = torch.zeros([output['action'].shape[0], self._cfg.original_action_shape])
                    if self._cuda:
                        output_action = to_device(output_action, self._device)
                    # the latent of extreme_action 
                    mask = output['action'].ge(self._cfg.latent_action_shape) & output['action'].le(self._cfg.model.action_shape)  # TODO

                    # the usual latent of vqvae learned action
                    output_action[~mask] = self._vqvae_model.decode({'quantized_index': output['action'].masked_select(~mask), 'obs': data.masked_select(~mask.unsqueeze(-1)).view(-1,self._cfg.model.obs_shape), 'threshold_phase': 'collect' in self._cfg.threshold_phase})[
                        'recons_action']

                    if mask.sum() > 0:
                        # the latent of extreme_action 
                        extreme_action_index = output['action'].masked_select(mask) - self._cfg.latent_action_shape
                        from itertools import product
                        # NOTE: disc_to_cont: transform discrete action index to original continuous action
                        self.m = self._cfg.original_action_shape
                        self.n = 2
                        self.K =  self.n ** self.m
                        self.disc_to_cont = list(product(*[list(range(self.n)) for dim in range(self.m)] ))
                        # NOTE: disc_to_cont: transform discrete action index to original continuous action
                        extreme_action = torch.tensor([[-1. if k==0 else 1.for k in  self.disc_to_cont[int(one_extreme_action_index)]] for one_extreme_action_index in extreme_action_index])
                        if self._cuda:
                            extreme_action = to_device(extreme_action, self._device)
                        output_action[mask] = extreme_action

                    output['action'] = output_action

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
