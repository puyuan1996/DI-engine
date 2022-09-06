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
        # return 'dqn', ['ding.model.template.q_learning']
        # return 'dqn_ma_share_backbone', ['ding.model.template.q_learning']
        return 'dqn_ma_share_backbone_head', ['ding.model.template.q_learning']


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

        if self._cfg.learn.rl_linear_lr_scheduler is True:
            from torch.optim.lr_scheduler import LambdaLR
            # rl_lambda = lambda step: (1e-5 / 3e-4 -1) * (1 / (3e6*20/256) ) * step + 1
            rl_lambda = lambda step: (1e-5 / 3e-4 -1) * (1 / (3e6 * self._cfg.learn.update_per_collect_rl/self._cfg.collect.n_sample) ) * step + 1
            self.rl_scheduler = LambdaLR(self._optimizer, lr_lambda=rl_lambda, last_epoch=-1)


        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        if not self._cfg.target_network_soft_update:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='target',
                update_type='assign',
                update_kwargs={'freq': self._cfg.learn.target_update_freq}
            )
        else:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='target',
                update_type='momentum',
                update_kwargs={'theta': self._cfg.learn.target_update_theta}
            )

        self._learn_model = model_wrap(self._model, wrapper_name='multi_average_argmax_sample')
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
            recons_loss_cont_weight = self._cfg.recons_loss_cont_weight,
            v_contrastive_regularization = self._cfg.v_contrastive_regularization,
            contrastive_regularization_loss_weight = self._cfg.contrastive_regularization_loss_weight
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
        # warmup phase: train VQVAE
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
        # VQVAE+RL phase
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
                    if self._cfg.v_contrastive_regularization:

                        # ====================
                        # Q-learning forward
                        # ====================
                        self._learn_model.train()
                        self._target_model.train()
                        # Target q value
                        with torch.no_grad():
                            # target_q_value_list: list{20}, each element is shape [B, A]
                            target_q_value_list = self._target_model.forward(data['next_obs'])

                        # get the average q value for each action, target_q_value_tensor shape: (B, A, E), B is batcn_size, A is action shape, E is ensemble num
                        target_q_value_tensor = torch.stack([target_q_value['logit'] for target_q_value in target_q_value_list], dim=-1)
                        mean_target_q_value = torch.mean(torch.mean(target_q_value_tensor, dim=-1), dim=-1)  # shape (B, )
                        # approximate target v value of next state
                        data['target_v_value'] = mean_target_q_value

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
                if self._cfg.v_contrastive_regularization:
                    loss_dict['contrastive_regularization_loss'] = result['contrastive_regularization_loss'].item()
                

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
                # Q-learning forward
                # ====================
                self._learn_model.train()
                self._target_model.train()

                loss = torch.tensor([0.])
                if self._cuda:
                    loss = to_device(loss, self._device)

                # Current q value (main model)
                q_value_list = self._learn_model.forward(data['obs'])['logit']
                # Target q value
                with torch.no_grad():
                    # target_q_value_list:list{20}, each element is shape [B,A]
                    target_q_value_list = self._target_model.forward(data['next_obs'])
                    # Max avarage q value action (main model)
                    target_q_action = self._learn_model.forward(data['next_obs'])['action']

                # get the average q value for each action,   target_q_value_tensor shape: (B,A,E), B is batcn_size, A is action shape, E is ensemble num
                target_q_value_tensor = torch.stack([target_q_value['logit'] for target_q_value in target_q_value_list], dim=-1)
                # mean_target_q_value = torch.mean(torch.mean(target_q_value_tensor, dim=-1), dim=-1)  # shape (B, )

                min_target_q_value = torch.min( target_q_value_tensor, dim=-1)[0]
                for agent in range(self._cfg.model.ensemble_num): 
                    # NOTE: RL learn policy in latent action space, so here using data['latent_action']
                    data_n = q_nstep_td_data(
                        q_value_list[agent]['logit'], min_target_q_value, data['latent_action'].squeeze(-1), target_q_action, data['reward'], data['done'], data['weight']
                    )
                    value_gamma = data.get('value_gamma')
                    loss_agent, td_error_per_sample_agent = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma)
                    loss += loss_agent
                    # loss_list.append(loss_agent)

                # TODO(pu): td3_bc loss
                if self._cfg.auxiliary_conservative_loss:
                    alpha=2.5
                    self.alpha = alpha
                    auxiliary_conservative_loss = q_value.mean()
                    # add behavior cloning loss weight(\lambda)
                    lmbda = self.alpha / q_value.abs().mean().detach()
                    loss = lmbda * auxiliary_conservative_loss + loss

                # ====================
                # Q-learning update
                # ====================
                self._optimizer.zero_grad()
                loss.backward()
                total_grad_norm_rl = self._optimizer.get_grad()
                if self._cfg.learn.multi_gpu:
                    self.sync_gradients(self._learn_model)
                self._optimizer.step()
                if self._cfg.learn.rl_linear_lr_scheduler is True:
                    self.rl_scheduler.step()

                # =============
                # after update
                # =============
                self._target_model.update(self._learn_model.state_dict())
                loss_dict['critic_loss'] = loss.item()

                q_value_dict = {}
                # get statistics for multi q value
                q_value_tensor = torch.stack([q_value['logit'] for q_value in q_value_list], dim=-1).detach().cpu() # shape (B, A, Ensemble_num)
                mean_q_value = torch.mean(q_value_tensor, dim=-1)  # shape (B, A)
                min_q_value = torch.min(q_value_tensor, dim=-1)[0]  # shape (B, A)
                max_q_value = torch.max(q_value_tensor, dim=-1)[0]  # shape (B, A)

                q_value_dict['mean_q_value'] =  mean_q_value.mean().item()
                q_value_dict['min_q_value'] =  min_q_value.mean().item()
                q_value_dict['max_q_value'] =  max_q_value.mean().item()
                if self._cfg.learn.rl_linear_lr_scheduler is True:
                    return {
                        # 'cur_lr': self._optimizer.defaults['lr'],
                        'cur_lr': self._optimizer.param_groups[0]['lr'],
                        'cur_scheduler_lr': self.rl_scheduler.get_last_lr()[0],
                        # 'td_error': td_error_per_sample.abs().mean(),
                        **loss_dict,
                        **q_value_dict,
                        'total_grad_norm_rl': total_grad_norm_rl,
                        # 'rewrad_run': data['rewrad_run'].mean().item(),
                        # 'rewrad_ctrl': data['rewrad_ctrl'].mean().item(),
                    }
                else:
                    return {
                        # 'cur_lr': self._optimizer.defaults['lr'],
                        'cur_lr': self._optimizer.param_groups[0]['lr'],
                        # 'td_error': td_error_per_sample.abs().mean(),
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
            'cur_scheduler_lr',
            'critic_loss',
            # 'q_value',
            'mean_q_value',
            'min_q_value',
            'max_q_value',
            # 'td_error',
            'total_vqvae_loss',
            'reconstruction_loss',
            'embedding_loss',
            'commitment_loss',
            'vq_loss',
            'total_grad_norm_rl',
            'total_grad_norm_vqvae',
            # mask statistics
            'recons_action_probs_left_mask_proportion',
            'recons_action_probs_right_mask_proportion',
            # 'predict_loss',
            # '[histogram]latent_action',
            # '[histogram]cos_similarity',
        ]
        if self._cfg.obs_regularization:
            ret.append('predict_loss')
        if self._cfg.v_contrastive_regularization:
            ret.append('contrastive_regularization_loss')
        return ret
        
    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``, initialize algorithm arguments and collect_model, \
            enable the eps_greedy_sample for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='multi_average_eps_greedy_sample')
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


            # debug
            # latents = to_device(torch.arange(64), 'cuda')
            # recons_action = self._vqvae_model.decode({'quantized_index': latents, 'obs': data, 'threshold_phase': False})['recons_action']
            # print(recons_action.max(0), recons_action.min(0),recons_action.mean(0), recons_action.std(0))

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
                # continuos action space
                if not self._cfg.augment_extreme_action:
                    # TODO
                    output['action'] = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data, 'threshold_phase': 'collect' in self._cfg.threshold_phase})[
                        'recons_action']
                    # output = self._vqvae_model.decode({'quantized_index': output['action'], 'obs': data})
                    # output['action'] = output['recons_action']
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

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='multi_average_argmax_sample')
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


    def visualize_latent(self, save_histogram=False, save_mapping=False, save_decoding_mapping=False, name_suffix=0, granularity=0.01, k=8):
        # i.e. to execute:
        # action_embedding = self._get_action_embedding(data)
        if self.cfg.action_space == 'continuous':
            # continuous action, for lunarlander env: 2 dim cont
            xx, yy = np.meshgrid(
                np.arange(-1, 1, granularity), np.arange(-1, 1, granularity)
            )
            cnt = int((2 / granularity)) ** 2
            action_samples = np.array([xx.ravel(), yy.ravel()]).reshape(cnt, 2)

            # continuous action, for hopper env: 3 dim cont
            # xx, yy, zz = np.meshgrid(
            #     np.arange(-1, 1, granularity), np.arange(-1, 1, granularity), np.arange(-1, 1, granularity)
            # )
            # cnt = int((2 / granularity)) ** 3
            # action_samples = np.array([xx.ravel(), yy.ravel(), zz.ravel()]).reshape(cnt, 3)
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
            encoding = self._vqvae_model.encoder(to_device(torch.Tensor(action_samples), self._device))
            quantized_index = self._vqvae_model.vq_layer.encode(encoding)

        if save_histogram:
            fig = plt.figure()

            # Fixing bin edges
            HIST_BINS = np.linspace(0, self._cfg.latent_action_shape - 1, self._cfg.latent_action_shape)

            # the histogram of the data
            n, bins, patches = plt.hist(
                quantized_index.detach().cpu().numpy(), HIST_BINS, density=False, facecolor='g', alpha=0.75
            )

            plt.xlabel('Latent Action')
            plt.ylabel('Count')
            plt.title('Latent Action Histogram')
            plt.grid(True)
            plt.show()

            if isinstance(name_suffix, int):
                plt.savefig(f'latent_histogram_{name_suffix}.png')
                print(f'save latent_histogram_{name_suffix}.png')
            elif isinstance(name_suffix, str):
                plt.savefig('latent_histogram_' + name_suffix + '.png')
                print('latent_histogram_' + name_suffix + '.png')
        elif save_mapping:
            xx, yy = np.meshgrid(np.arange(-1, 1, granularity), np.arange(-1, 1, granularity))

            x = xx.ravel()
            y = yy.ravel()
            c = quantized_index.detach().cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sc = ax.scatter(x, y, c=c, marker='o')
            plt.xlabel('Original Action Dim0')
            plt.ylabel('Original Action Dim1')
            ax.set_title('Latent Action Mapping')
            fig.colorbar(sc)
            plt.show()
            plt.savefig(f'latent_mapping_{name_suffix}.png')
        elif save_decoding_mapping:
            # TODO: k
            latents = to_device(torch.arange(k), 'cuda')
            
            # if obs-conditioned
            obs = torch.tensor([0,  1.4135e+00, -5.9936e-02,  1.1277e-01,  6.9229e-04, 1.3576e-02,  0.0000e+00,  0.0000e+00])
            # obs = torch.tensor([-1,  1.4135e+00, -5.9936e-02,  1.1277e-01,  6.9229e-04, 1.3576e-02,  0.0000e+00,  0.0000e+00])
            obs =  obs.repeat(8,1)
            obs = to_device( obs, 'cuda')
            recons_action = self._vqvae_model.decode({'quantized_index': latents, 'obs': obs, 'threshold_phase': False})['recons_action'].detach().cpu().numpy()
            
            # if no obs-conditioned
            # recons_action = self._vqvae_model.decode({'quantized_index': latents, 'obs': None, 'threshold_phase': False})['recons_action'].detach().cpu().numpy()
            
            print(recons_action.max(0), recons_action.min(0),recons_action.mean(0), recons_action.std(0))

            
            c = latents.detach().cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot(111)

            sc = ax.scatter(recons_action[:,0], recons_action[:,1], c=c, marker='o')
            
            # annotaions
            annotations=[f"{i}" for i in range(k)]
            for i, label in enumerate(annotations):
                plt.text(recons_action[:,0][i], recons_action[:,1][i],label)

            plt.xlabel('Original Action Dim0')
            plt.ylabel('Original Action Dim1')
            ax.set_title('Latent Action Decoding')
            fig.colorbar(sc)
            #设置坐标轴范围
            plt.xlim((-1, 1))
            plt.ylim((-1, 1))
            plt.show()
            plt.savefig(f'latent_action_decoding_{name_suffix}.png')
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


