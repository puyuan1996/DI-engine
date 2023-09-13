import copy
from typing import List

import numpy as np
import gym
from gym.spaces import Box
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY

from ding.torch_utils import to_ndarray, to_list

@ENV_REGISTRY.register('toy_env')
class ToyEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self.act_transform = cfg.act_transform
        cfg.max_episode_length = 10
        self.max_episode_length = cfg.max_episode_length
        self.sparse_reward_setting = True
        self._state = 0
        self._observation_space = Box(low=np.zeros(4), high=np.ones(4), dtype=np.float32)
        self._action_space = Box(low=np.array([0., 0.]), high=np.array([4, 4]), dtype=np.float32)
        self._reward_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self._states_action_spaces = [
            Box(low=np.array([0.75, 0.75]), high=np.array([1.125, 1.125]), dtype=np.float32),
            Box(low=np.array([2.875, 0.75]), high=np.array([3.25, 1.125]), dtype=np.float32),
            Box(low=np.array([0.75, 2.875]), high=np.array([1.125, 3.25]), dtype=np.float32),
            Box(low=np.array([2.875, 2.875]), high=np.array([3.25, 3.25]), dtype=np.float32),
        ]

    def reset(self, init_state: np.ndarray = None) -> np.ndarray:
        # TODO(pu)
        if init_state is None:
            self._state = np.random.choice([0, 1, 2, 3])
            self._state = np.array([0])
        else:
            self._state = init_state
        self._state_action_space = self._states_action_spaces[self._state[0]]
        obs = self._one_hot_state()
        obs = to_ndarray(obs).astype(np.float32)
        self._eval_episode_return = 0.
        self.timestep = 0
        return obs

    def close(self) -> None:
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        np.random.seed(seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        # ppo (mu,sigma)
        # when eval, action = mu must be in [-1, 1]
        # but when collect, sample action from (mu, sigma), action is in [-inf, inf]
        # print(f"input action: {action}")

        # action = to_ndarray(action).astype(np.float32)

        if self.act_transform:
            # action each element is in [-1, 1], scale into [0, 4]
            action = (action + 1) * 2

        # print(f"before clip: {action}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # print(f"after clip: {action}")

        self._state_action_space = self._states_action_spaces[self._state[0]]
        if (self._state_action_space.high[1] - self._state_action_space.low[1]) / 2 + self._state_action_space.low[1] < \
                action[1] < self._state_action_space.high[1]:
            self._state = (self._state + 1) % 4

        if self.sparse_reward_setting:
            # sparse_reward setting
            # the env_steps for ppo converge:
            # entropy_weight = 0.0, 65k converge
            # entropy_weight = 0.001, 50k converge
            reward = 1 if self._state == 3 else 0
        else:
            # dense reward setting
            # the env_steps for ppo converge:
            # entropy_weight = 0.0, 32k converge
            # entropy_weight = 0.001, 48k converge
            if self._state == 0:
                reward = 0
            elif self._state == 1:
                reward = 0.01
            elif self._state == 2:
                reward = 0.01
            elif self._state == 3:
                reward = 1

        self._eval_episode_return += reward
        obs = self._one_hot_state()
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([reward]).astype(np.float32)

        print(f'obs: {obs}')
        print((obs == np.array([0, 0, 0, 1])).all())
        done = bool((obs == np.array([0, 0, 0, 1])).all())

        print('done:', done, 'reward:', reward, 'state:', self._state, 'action:', action, 'timestep:', self.timestep)

        """
        done: False reward: 0 state: [2] action: [0.8928997 3.1194117] timestep: 6
        done: False reward: 0 state: [2] action: [0.8928997 3.1194117] timestep: 7
        done: False reward: 0 state: [2] action: [0.8928997 3.1194117] timestep: 8
        done: False reward: 0 state: [2] action: [0.8928997 3.1194117] timestep: 9
        """
        info = {}
        self.timestep += 1
        if self.timestep >= self.max_episode_length:
            done = True
        if done:
            info['eval_episode_return'] = self._eval_episode_return


        return BaseEnvTimestep(obs, rew, done, info)

    def _one_hot_state(self) -> np.ndarray:
        one_hot_state = np.zeros(4)
        one_hot_state[self._state] = 1
        return one_hot_state

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Toy Env"