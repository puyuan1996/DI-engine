from typing import List, Dict, Any, Optional, Tuple, Union
from collections import namedtuple, defaultdict
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.policy import Policy
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, DatasetNormalizer
from ding.utils.data import default_collate, default_decollate
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('pd')
class PDPolicy(Policy):
    r"""
    Overview:
        Implicit Plan Diffuser
        https://arxiv.org/pdf/2205.09991.pdf

    """
    config = dict(
        type='pd',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Default False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 10000 in SAC.
        random_collect_size=10000,
        nstep=1,
        # normalizer type
        normalizer='GaussianNormalizer',
        model=dict(
            diffuser_model='GaussianDiffusion',
            diffuser_model_cfg=dict(
                # the type of model
                model='TemporalUnet',
                # config of model
                model_cfg=dict(
                    # model dim, In GaussianInvDynDiffusion, it is obs_dim. In others, it is obs_dim + action_dim
                    transition_dim=23,
                    dim=32,
                    dim_mults=[1, 2, 4, 8],
                    # whether use return as a condition
                    returns_condition=False,
                    condition_dropout=0.1,
                    # whether use calc energy
                    calc_energy=False,
                    kernel_size=5,
                    # whether use attention
                    attention=False,
                ),
                # horizon of tarjectory which generated by model
                horizon=80,
                # timesteps of diffusion
                n_timesteps=1000,
                # hidden dim of action model
                # Whether predict epsilon
                predict_epsilon=True,
                # discount of loss
                loss_discount=1.0,
                # whether clip denoise
                clip_denoised=False,
                action_weight=10,
            ),
            value_model='ValueDiffusion',
            value_model_cfg=dict(
                # the type of model
                model='TemporalValue',
                # config of model
                model_cfg=dict(
                    horizon=4,
                    # model dim, In GaussianInvDynDiffusion, it is obs_dim. In others, it is obs_dim + action_dim
                    transition_dim=23,
                    dim=32,
                    dim_mults=[1, 2, 4, 8],
                    # whether use calc energy
                    kernel_size=5,
                ),
                # horizon of tarjectory which generated by model
                horizon=80,
                # timesteps of diffusion
                n_timesteps=1000,
                # hidden dim of action model
                predict_epsilon=True,
                # discount of loss
                loss_discount=1.0,
                # whether clip denoise
                clip_denoised=False,
                action_weight=1.0,
            ),
            # guide_steps for p sample
            n_guide_steps=2,
            # scale of grad for p sample
            scale=0.1,
            # t of stopgrad for p sample
            t_stopgrad=2,
            # whether use std as a scale for grad
            scale_grad_by_std=True,
        ),
        learn=dict(

            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=100,

            # (float type) learning_rate_q: Learning rate for model.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate=3e-4,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,

            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            gradient_accumulate_every=2,
            # train_epoch = train_epoch * gradient_accumulate_every
            train_epoch=60000,
            # batch_size of every env when eval
            plan_batch_size=64,

            # step start update target model and frequence
            step_start_update_target=2000,
            update_target_freq=10,
            # update weight of target net
            target_weight=0.995,
            value_step=200e3,

            # dataset weight include returns
            include_returns=True,

            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'pd', ['ding.model.template.diffusion']

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self.action_dim = self._cfg.model.diffuser_model_cfg.action_dim
        self.obs_dim = self._cfg.model.diffuser_model_cfg.obs_dim
        self.n_timesteps = self._cfg.model.diffuser_model_cfg.n_timesteps
        self.gradient_accumulate_every = self._cfg.learn.gradient_accumulate_every
        self.plan_batch_size = self._cfg.learn.plan_batch_size
        self.gradient_steps = 1
        self.update_target_freq = self._cfg.learn.update_target_freq
        self.step_start_update_target = self._cfg.learn.step_start_update_target
        self.target_weight = self._cfg.learn.target_weight
        self.value_step = self._cfg.learn.value_step
        # self.use_target = True
        self.use_target = False
        self.horizon = self._cfg.model.diffuser_model_cfg.horizon
        self.include_returns = self._cfg.learn.include_returns

        # Optimizers
        self._plan_optimizer = Adam(
            self._model.diffuser.model.parameters(),
            lr=self._cfg.learn.learning_rate,
        )
        if self._model.value:
            self._value_optimizer = Adam(
                self._model.value.model.parameters(),
                lr=self._cfg.learn.learning_rate,
            )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        # self._target_model = model_wrap(
        #     self._target_model,
        #     wrapper_name='target',
        #     update_type='momentum',
        #     update_kwargs={'theta': self._cfg.learn.target_theta}
        # )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        # self._target_model.reset()

        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        loss_dict = {}

        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )

        conds = {}
        vals = data['condition_val']
        ids = data['condition_id']
        for i in range(len(ids)):
            conds[ids[i][0].item()] = vals[i]
        if len(ids) > 1:
            self.use_target = True
        data['conditions'] = conds
        if 'returns' in data.keys():
            data['returns'] = data['returns'].unsqueeze(-1)
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        # self._target_model.train()
        x = data['trajectories']

        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size, ), device=x.device).long()
        cond = data['conditions']
        if 'returns' in data.keys():
            target = data['returns']
        loss_dict['diffuse_loss'], loss_dict['a0_loss'] = self._model.diffuser_loss(x, cond, t)
        loss_dict['diffuse_loss'] = loss_dict['diffuse_loss'] / self.gradient_accumulate_every
        loss_dict['diffuse_loss'].backward()
        if self._forward_learn_cnt < self.value_step and self._model.value:
            loss_dict['value_loss'], logs = self._model.value_loss(x, cond, target, t)
            loss_dict['value_loss'] = loss_dict['value_loss'] / self.gradient_accumulate_every
            loss_dict['value_loss'].backward()
            loss_dict.update(logs)

        if self.gradient_steps >= self.gradient_accumulate_every:
            self._plan_optimizer.step()
            self._plan_optimizer.zero_grad()
            if self._forward_learn_cnt < self.value_step and self._model.value:
                self._value_optimizer.step()
                self._value_optimizer.zero_grad()
            self.gradient_steps = 1
        else:
            self.gradient_steps += 1
        self._forward_learn_cnt += 1
        if self._forward_learn_cnt % self.update_target_freq == 0:
            if self._forward_learn_cnt < self.step_start_update_target:
                self._target_model.load_state_dict(self._model.state_dict())
            else:
                self.update_model_average(self._target_model, self._learn_model)

        if 'returns' in data.keys():
            loss_dict['max_return'] = target.max().item()
            loss_dict['min_return'] = target.min().item()
            loss_dict['mean_return'] = target.mean().item()
        loss_dict['max_traj'] = x.max().item()
        loss_dict['min_traj'] = x.min().item()
        loss_dict['mean_traj'] = x.mean().item()
        return loss_dict

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            if old_weight is None:
                ma_params.data = up_weight
            else:
                old_weight * self.target_weight + (1 - self.target_weight) * up_weight

    def _monitor_vars_learn(self) -> List[str]:
        return [
            'diffuse_loss',
            'value_loss',
            'max_return',
            'min_return',
            'mean_return',
            'max_traj',
            'min_traj',
            'mean_traj',
            'mean_pred',
            'max_pred',
            'min_pred',
            'a0_loss',
        ]

    def _state_dict_learn(self) -> Dict[str, Any]:
        if self._model.value:
            return {
                'model': self._learn_model.state_dict(),
                'target_model': self._target_model.state_dict(),
                'plan_optimizer': self._plan_optimizer.state_dict(),
                'value_optimizer': self._value_optimizer.state_dict(),
            }
        else:
            return {
                'model': self._learn_model.state_dict(),
                'target_model': self._target_model.state_dict(),
                'plan_optimizer': self._plan_optimizer.state_dict(),
            }

    def _init_eval(self):
        self._eval_model = model_wrap(self._target_model, wrapper_name='base')
        self._eval_model.reset()
        if self.use_target:
            self._plan_seq = []

    def init_data_normalizer(self, normalizer: DatasetNormalizer = None):
        self.normalizer = normalizer

    def _forward_eval(self, data: dict) -> Dict[str, Any]:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))

        self._eval_model.eval()
        if self.use_target:
            cur_obs = self.normalizer.normalize(data[:, :self.obs_dim], 'observations')
            target_obs = self.normalizer.normalize(data[:, self.obs_dim:], 'observations')
        else:
            obs = self.normalizer.normalize(data, 'observations')
        with torch.no_grad():
            if self.use_target:
                cur_obs = torch.tensor(cur_obs)
                target_obs = torch.tensor(target_obs)
                if self._cuda:
                    cur_obs = to_device(cur_obs, self._device)
                    target_obs = to_device(target_obs, self._device)
                conditions = {0: cur_obs, self.horizon - 1: target_obs}
            else:
                obs = torch.tensor(obs)
                if self._cuda:
                    obs = to_device(obs, self._device)
                conditions = {0: obs}

            if self.use_target:
                if self._plan_seq == [] or 0 in self._eval_t:
                    plan_traj = self._eval_model.get_eval(conditions, self.plan_batch_size)
                    plan_traj = to_device(plan_traj, 'cpu').numpy()
                    if self._plan_seq == []:
                        self._plan_seq = plan_traj
                        self._eval_t = [0] * len(data_id)
                    else:
                        for id in data_id:
                            if self._eval_t[id] == 0:
                                self._plan_seq[id] = plan_traj[id]
                action = []
                for id in data_id:
                    if self._eval_t[id] < len(self._plan_seq[id]) - 1:
                        next_waypoint = self._plan_seq[id][self._eval_t[id] + 1]
                    else:
                        next_waypoint = self._plan_seq[id][-1].copy()
                        next_waypoint[2:] = 0
                    cur_ob = cur_obs[id]
                    cur_ob = to_device(cur_ob, 'cpu').numpy()
                    act = next_waypoint[:2] - cur_ob[:2] + (next_waypoint[2:] - cur_ob[2:])
                    action.append(act)
                    self._eval_t[id] += 1
            else:
                action = self._eval_model.get_eval(conditions, self.plan_batch_size)
                if self._cuda:
                    action = to_device(action, 'cpu')
                action = self.normalizer.unnormalize(action, 'actions')
            action = torch.tensor(action).to('cpu')
        output = {'action': action}
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        if self.use_target and data_id:
            for id in data_id:
                self._eval_t[id] = 0

    def _init_collect(self) -> None:
        pass

    def _forward_collect(self, data: dict, **kwargs) -> dict:
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        pass

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        pass
