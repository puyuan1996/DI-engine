from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import torch

from ding.torch_utils import Adam, to_device, to_tensor, unsqueeze
from ding.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, get_gae_with_default_last_value, \
    v_nstep_td_data, v_nstep_td_error, get_nstep_return_data, get_train_sample, gae, gae_data, ppo_error_continuous, \
    get_gae
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, split_data_generator, RunningMeanStd, dicts_to_lists, lists_to_dicts
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn
from ding.model.template.action_vqvae import ActionVQVAE

from ding.utils import RunningMeanStd
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from ding.envs.common import affine_transform
from torch.distributions import Independent, Normal


@POLICY_REGISTRY.register('onppo-vqvae')
class ONPPOVQVAEPolicy(Policy):
    r"""
    Overview:
        Policy class of onppo-VQVAE algorithm.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      onppo            | RL policy register name, refer to      | This arg is optional,
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
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='onppo-vqvae',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        # (bool) Whether to use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update due to priority.
        # If True, priority must be True.
        priority_IS_weight=False,
        # (bool) Whether to recompurete advantages in each iteration of on-policy PPO
        recompute_adv=True,
        # (str) Which kind of action space used in PPOPolicy, ['discrete', 'continuous', 'hybrid']
        action_space='discrete',
        # (bool) Whether to use nstep return to calculate value target, otherwise, use return = adv + value
        nstep_return=False,
        # (bool) Whether to enable multi-agent training, i.e.: MAPPO
        multi_agent=False,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=True,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.0,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=True,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
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
    
    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For onppo, ``ding.model.template.q_learning.onppo``
        """
        if self._cfg.multi_agent:
            return 'mavac', ['ding.model.template.mavac']
        else:
            return 'vac', ['ding.model.template.vac']

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main model.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in PPO"

        self._action_space = self._cfg.action_space
        if self._cfg.learn.ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            if self._action_space in ['continuous', 'hybrid']:
                # init log sigma
                if self._action_space == 'continuous':
                    if hasattr(self._model.actor_head, 'log_sigma_param'):
                        torch.nn.init.constant_(self._model.actor_head.log_sigma_param, -0.5)
                elif self._action_space == 'hybrid':  # actor_head[1]: ReparameterizationHead, for action_args
                    if hasattr(self._model.actor_head[1], 'log_sigma_param'):
                        torch.nn.init.constant_(self._model.actor_head[1].log_sigma_param, -0.5)

                for m in list(self._model.critic.modules()) + list(self._model.actor.modules()):
                    if isinstance(m, torch.nn.Linear):
                        # orthogonal initialization
                        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        torch.nn.init.zeros_(m.bias)
                # do last policy layer scaling, this will make initial actions have (close to)
                # 0 mean and std, and will help boost performances,
                # see https://arxiv.org/abs/2006.05990, Fig.24 for details
                for m in self._model.actor.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.zeros_(m.bias)
                        m.weight.data.copy_(0.01 * m.weight.data)

        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.grad_clip_value
        )

        self._learn_model = model_wrap(self._model, wrapper_name='base')

        # Algorithm config
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm
        self._value_norm = self._cfg.learn.value_norm
        if self._value_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv
        # Main model
        self._learn_model.reset()

        # vqvae related
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
            return_infos = []
            self._forward_learn_cnt += 1
            loss_dict = {}
            q_value_dict = {}
            data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
            if self._cuda:
                data = to_device(data, self._device)
            # ====================
            # train VQVAE
            # ====================
            if data['vae_phase'][0].item() is True:
                for epoch in range(self._cfg.learn.epoch_per_collect_vqvae):
                    for batch_data in split_data_generator(data, self._cfg.learn.vqvae_batch_size, shuffle=True):
                        
                        if self._cfg.obs_regularization:
                            batch_data['true_residual'] = batch_data['next_obs'] - batch_data['obs']
                            result = self._vqvae_model.train_with_obs(batch_data)
                        else:
                            result = self._vqvae_model.train(batch_data)

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

                        return_info= {
                            'priority': return_normalization.tolist(), 
                            'cur_lr': self._optimizer.defaults['lr'],
                            # 'td_error': td_error_per_sample,
                            **loss_dict,
                            # **q_value_dict,
                            'total_grad_norm_vqvae': total_grad_norm_vqvae,
                            # '[histogram]latent_action': quantized_index,
                            # '[histogram]cos_similarity': cos_similarity,
                        }
                        return_infos.append(return_info)
                return return_infos
            # ====================
            # train RL
            # ====================
            else:



                # ====================
                # PPO forward
                # ====================
                return_infos = []
                self._learn_model.train()

                for epoch in range(self._cfg.learn.epoch_per_collect_rl):
                    if self._recompute_adv:  # new v network compute new value
                        # # TODO
                        # # Representation shift correction (RSC)
                        # if self._cfg.recompute_latent_action:
                        #     quantized_index = self._vqvae_model.encode({'action': data['action']})
                        #     data['latent_action'] = quantized_index.clone().detach()
                        #     # print(torch.unique(data['latent_action']))
                        
                        # NOTE: RL learn policy in latent action space, so here using data['latent_action']
                        data['action'] = copy.deepcopy(data['latent_action'])

                        with torch.no_grad():
                            value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                            next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                            if self._value_norm:
                                value *= self._running_mean_std.std
                                next_value *= self._running_mean_std.std

                            traj_flag = data.get('traj_flag', None)  # traj_flag indicates termination of trajectory
                            compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], traj_flag)
                            data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)

                            unnormalized_returns = value + data['adv']

                            if self._value_norm:
                                data['value'] = value / self._running_mean_std.std
                                data['return'] = unnormalized_returns / self._running_mean_std.std
                                self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                            else:
                                data['value'] = value
                                data['return'] = unnormalized_returns

                    else:  # don't recompute adv
                        if self._value_norm:
                            unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                            data['return'] = unnormalized_return / self._running_mean_std.std
                            self._running_mean_std.update(unnormalized_return.cpu().numpy())
                        else:
                            data['return'] = data['adv'] + data['value']

                    for batch in split_data_generator(data, self._cfg.learn.rl_batch_size, shuffle=True):
                        output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')
                        adv = batch['adv']
                        if self._adv_norm:
                            # Normalize advantage in a train_batch
                            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                        # Calculate ppo error
                        if self._action_space == 'continuous':
                            ppo_batch = ppo_data(
                                output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                                batch['return'], batch['weight']
                            )
                            ppo_loss, ppo_info = ppo_error_continuous(ppo_batch, self._clip_ratio)
                        elif self._action_space == 'discrete':
                            ppo_batch = ppo_data(
                                output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                                batch['return'], batch['weight']
                            )
                            ppo_loss, ppo_info = ppo_error(ppo_batch, self._clip_ratio)
                        elif self._action_space == 'hybrid':
                            # discrete part (discrete policy loss and entropy loss)
                            ppo_discrete_batch = ppo_policy_data(
                                output['logit']['action_type'], batch['logit']['action_type'], batch['action']['action_type'],
                                adv, batch['weight']
                            )
                            ppo_discrete_loss, ppo_discrete_info = ppo_policy_error(ppo_discrete_batch, self._clip_ratio)
                            # continuous part (continuous policy loss and entropy loss, value loss)
                            ppo_continuous_batch = ppo_data(
                                output['logit']['action_args'], batch['logit']['action_args'], batch['action']['action_args'],
                                output['value'], batch['value'], adv, batch['return'], batch['weight']
                            )
                            ppo_continuous_loss, ppo_continuous_info = ppo_error_continuous(
                                ppo_continuous_batch, self._clip_ratio
                            )
                            # sum discrete and continuous loss
                            ppo_loss = type(ppo_continuous_loss)(
                                ppo_continuous_loss.policy_loss + ppo_discrete_loss.policy_loss, ppo_continuous_loss.value_loss,
                                ppo_continuous_loss.entropy_loss + ppo_discrete_loss.entropy_loss
                            )
                            ppo_info = type(ppo_continuous_info)(
                                max(ppo_continuous_info.approx_kl, ppo_discrete_info.approx_kl),
                                max(ppo_continuous_info.clipfrac, ppo_discrete_info.clipfrac)
                            )
                        wv, we = self._value_weight, self._entropy_weight
                        total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

                        self._optimizer.zero_grad()
                        total_loss.backward()
                        total_grad_norm_rl = self._optimizer.get_grad()
                        self._optimizer.step()

                        return_info = {
                            'cur_lr': self._optimizer.defaults['lr'],
                            'total_loss': total_loss.item(),
                            'policy_loss': ppo_loss.policy_loss.item(),
                            'value_loss': ppo_loss.value_loss.item(),
                            'entropy_loss': ppo_loss.entropy_loss.item(),
                            'adv_max': adv.max().item(),
                            'adv_mean': adv.mean().item(),
                            'value_mean': output['value'].mean().item(),
                            'value_max': output['value'].max().item(),
                            'approx_kl': ppo_info.approx_kl,
                            'clipfrac': ppo_info.clipfrac,
                            'total_grad_norm_rl': total_grad_norm_rl,
                        }
                        if self._action_space == 'continuous':
                            return_info.update(
                                {
                                    'act': batch['action'].float().mean().item(),
                                    'mu_mean': output['logit']['mu'].mean().item(),
                                    'sigma_mean': output['logit']['sigma'].mean().item(),
                                }
                            )
                        return_infos.append(return_info)
                return return_infos


    def _monitor_vars_learn(self) -> List[str]:
        variables = super()._monitor_vars_learn() + [
            'policy_loss',
            'value_loss',
            'entropy_loss',
            'adv_max',
            'adv_mean',
            'approx_kl',
            'clipfrac',
            'value_max',
            'value_mean',
            # vqvae related
            'total_vqvae_loss',
            'reconstruction_loss',
            'embedding_loss',
            'commitment_loss',
            'vq_loss',
            'total_grad_norm_rl',
            'total_grad_norm_vqvae',
        ]
        if self._action_space == 'continuous':
            variables += ['mu_mean', 'sigma_mean', 'sigma_grad', 'act']
        return variables

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            # 'target_model': self._target_model.state_dict(),
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
        # self._target_model.load_state_dict(state_dict['target_model'])
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
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._collect_model = model_wrap(self._model, wrapper_name='reparam_sample')
        elif self._action_space == 'discrete':
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        elif self._action_space == 'hybrid':
            self._collect_model = model_wrap(self._model, wrapper_name='hybrid_reparam_multinomial_sample')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv
        self._warm_up_stop = False

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
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
            # output = self._collect_model.forward(data, eps=eps)
            output = self._collect_model.forward(data, mode='compute_actor_critic')
            # here output['action'] is the out of onppo, is discrete action
            output['latent_action'] = copy.deepcopy(output['action'])
            if self._cuda:
                output = to_device(output, self._device)

            # TODO(pu): decode into original hybrid actions, here data is obs
            # this is very important to generate self.obs_encoding using in decode phase
            # output['action'] = self._vqvae_model.decode_with_obs(output['action'], data})['recons_action']

            if self._cfg.original_action_space == 'hybrid':
                # TODO(pu): decode into original hybrid actions, here data is obs
                recons_action = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data})
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
                # continous action space
                if not self._cfg.augment_extreme_action:
                    output['action'] = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data})[
                        'recons_action']
                else:
                    output_action = torch.zeros([output['action'].shape[0], self._cfg.original_action_shape])
                    if self._cuda:
                        output_action = to_device(output_action, self._device)
                    # the latent of extreme_action, e.g. [64, 64+2**3) [64, 72)
                    mask = output['action'].ge(self._cfg.latent_action_shape) & output['action'].le(self._cfg.model.action_shape)  # TODO

                    # the usual latent of vqvae learned action
                    output_action[~mask] = self._vqvae_model.decode({'quantized_index': output['action'].masked_select(~mask), 'obs': data.masked_select(~mask.unsqueeze(-1)).view(-1,self._cfg.model.obs_shape)})[
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

    # def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and calculate GAE, return one data to cache for next time calculation
        Arguments:
            - data (:obj:`list`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        # for item in data:
        #     item['return'] = torch.stack([data[i]['reward'] for i in range(len(data))]).sum(0)
          
        data = to_device(data, self._device)
        for transition in data:
            transition['traj_flag'] = copy.deepcopy(transition['done'])
        data[-1]['traj_flag'] = True

        if self._cfg.learn.ignore_done:
            data[-1]['done'] = False

        if data[-1]['done']:
            last_value = torch.zeros_like(data[-1]['value'])
        else:
            with torch.no_grad():
                last_value = self._collect_model.forward(
                    unsqueeze(data[-1]['next_obs'], 0), mode='compute_actor_critic'
                )['value']
            if len(last_value.shape) == 2:  # multi_agent case:
                last_value = last_value.squeeze(0)
        if self._value_norm:
            last_value *= self._running_mean_std.std
            for i in range(len(data)):
                data[i]['value'] *= self._running_mean_std.std
        data = get_gae(
            data,
            to_device(last_value, self._device),
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=False,
        )
        if self._value_norm:
            for i in range(len(data)):
                data[i]['value'] /= self._running_mean_std.std

        # remove next_obs for save memory when not recompute adv
        if not self._recompute_adv:
            for i in range(len(data)):
                data[i].pop('next_obs')
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        """
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation
                - model_output (:obj:`dict`): Output of collect model, including at least ['action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data.
        """

        if 'latent_action' in model_output.keys():
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': model_output['action'],
                'logit': model_output['logit'],
                'value': model_output['value'],
                'latent_action': model_output['latent_action'],
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
                'action': model_output['action'],
                'logit': model_output['logit'],
                'value': model_output['value'],
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
        self._action_space = self._cfg.action_space
        if self._action_space == 'continuous':
            self._eval_model = model_wrap(self._model, wrapper_name='deterministic_sample')
        elif self._action_space == 'discrete':
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        elif self._action_space == 'hybrid':
            self._eval_model = model_wrap(self._model, wrapper_name='hybrid_deterministic_argmax_sample')
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
            # output = self._eval_model.forward(data)
            output = self._eval_model.forward(data, mode='compute_actor')

            # here output['action'] is the out of onppo, is discrete action
            # output['latent_action'] = output['action']  # TODO(pu)
            # print('k', output['action'])
            output['latent_action'] = copy.deepcopy(output['action'])

            # TODO(pu): decode into original hybrid actions, here data is obs
            # this is very important to generate self.obs_encoding using in decode phase
            # output['action'] = self._vqvae_model.decode_with_obs(output['action'], data})['recons_action']

            if self._cfg.original_action_space == 'hybrid':
                recons_action = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data})
                output['action'] = {
                    'action_type': recons_action['recons_action']['action_type'],
                    'action_args': recons_action['recons_action']['action_args']
                }
            else:
                # continuous action space
                if not self._cfg.augment_extreme_action:
                    output['action'] = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data})[
                        'recons_action']
                else:
                    output_action = torch.zeros([output['action'].shape[0], self._cfg.original_action_shape])
                    if self._cuda:
                        output_action = to_device(output_action, self._device)
                    # the latent of extreme_action 
                    mask = output['action'].ge(self._cfg.latent_action_shape) & output['action'].le(self._cfg.model.action_shape)  # TODO

                    # the usual latent of vqvae learned action
                    output_action[~mask] = self._vqvae_model.decode({'quantized_index': output['action'].masked_select(~mask), 'obs': data.masked_select(~mask.unsqueeze(-1)).view(-1,self._cfg.model.obs_shape)})[
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

    def visualize_latent(self, save_histogram=True, name=0, granularity=0.1):
        if self.cfg.original_action_space == 'continuous':
            # continuous action, now only for hopper env: 3 dim cont
            xx, yy, zz = np.meshgrid(
                np.arange(-1, 1, granularity), np.arange(-1, 1, granularity), np.arange(-1, 1, granularity)
            )
            cnt = int((2 / granularity)) ** 3
            action_samples = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).reshape(cnt, 3)
        elif self.cfg.original_action_space == 'hybrid':  
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

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Hopper-v3 onppo episode0_latent_actions')
# plt.plot(episode0_latent_actions)
# plt.show()
# plt.savefig(f'hopper-v3_onppo_episode0_latent_actions.png')
