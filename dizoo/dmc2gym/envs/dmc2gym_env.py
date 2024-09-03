import copy
import os
import os
from datetime import datetime
from typing import Callable, Union, Dict
from typing import Optional
from typing import Optional, Callable

import dmc2gym
import dmc2gym
import gym
import gym
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import WarpFrameWrapper, ScaledFloatFrameWrapper, ClipRewardWrapper, ActionRepeatWrapper, \
    FrameStackWrapper
from ding.envs import WarpFrameWrapper, ScaledFloatFrameWrapper, ClipRewardWrapper, ActionRepeatWrapper, \
    FrameStackWrapper
from ding.envs.common.common_function import affine_transform
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gym.spaces import Box
from gym.spaces import Box
from matplotlib import animation


def dmc2gym_observation_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Callable:

    def observation_space(from_pixels=True, height=84, width=84, channels_first=True) -> Box:
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            return Box(low=0, high=255, shape=shape, dtype=np.uint8)
        else:
            return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)

    return observation_space


def dmc2gym_state_space(dim, minimum=-np.inf, maximum=np.inf, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)


def dmc2gym_action_space(dim, minimum=-1, maximum=1, dtype=np.float32) -> Box:
    return Box(np.repeat(minimum, dim).astype(dtype), np.repeat(maximum, dim).astype(dtype), dtype=dtype)


def dmc2gym_reward_space(minimum=0, maximum=1, dtype=np.float32) -> Callable:

    def reward_space(frame_skip=1) -> Box:
        return Box(
            np.repeat(minimum * frame_skip, 1).astype(dtype),
            np.repeat(maximum * frame_skip, 1).astype(dtype),
            dtype=dtype
        )

    return reward_space


"""
default observation, state, action, reward space for dmc2gym env
"""
dmc2gym_env_info = {
    "ball_in_cup": {
        "catch": {
            "observation_space": dmc2gym_observation_space(8),
            "state_space": dmc2gym_state_space(8),
            "action_space": dmc2gym_action_space(2),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "cartpole": {
        "balance": {
            "observation_space": dmc2gym_observation_space(5),
            "state_space": dmc2gym_state_space(5),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        },
        "swingup": {
            "observation_space": dmc2gym_observation_space(5),
            "state_space": dmc2gym_state_space(5),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "cheetah": {
        "run": {
            "observation_space": dmc2gym_observation_space(17),
            "state_space": dmc2gym_state_space(17),
            "action_space": dmc2gym_action_space(6),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "finger": {
        "spin": {
            "observation_space": dmc2gym_observation_space(9),
            "state_space": dmc2gym_state_space(9),
            "action_space": dmc2gym_action_space(1),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "reacher": {
        "easy": {
            "observation_space": dmc2gym_observation_space(6),
            "state_space": dmc2gym_state_space(6),
            "action_space": dmc2gym_action_space(2),
            "reward_space": dmc2gym_reward_space()
        }
    },
    "walker": {
        "walk": {
            "observation_space": dmc2gym_observation_space(24),
            "state_space": dmc2gym_state_space(24),
            "action_space": dmc2gym_action_space(6),
            "reward_space": dmc2gym_reward_space()
        }
    }
}


@ENV_REGISTRY.register('dmc2gym')
class DMC2GymEnv(BaseEnv):

    def __init__(self, cfg: dict = {}) -> None:
        assert cfg.domain_name in dmc2gym_env_info, '{}/{}'.format(cfg.domain_name, dmc2gym_env_info.keys())
        assert cfg.task_name in dmc2gym_env_info[
            cfg.domain_name], '{}/{}'.format(cfg.task_name, dmc2gym_env_info[cfg.domain_name].keys())

        # default config for dmc2gym env
        self._cfg = {
            "frame_skip": 4,
            'warp_frame': False,
            'scale': False,
            'clip_rewards': False,
            'action_repeat': 1,
            "frame_stack": 3,
            "from_pixels": True,
            "visualize_reward": False,
            "height": 84,
            "width": 84,
            "channels_first": True,
            "resize": 84,
        }

        self._cfg.update(cfg)

        self._init_flag = False

        self._replay_path = None

        self._observation_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["observation_space"](
            from_pixels=self._cfg["from_pixels"],
            height=self._cfg["height"],
            width=self._cfg["width"],
            channels_first=self._cfg["channels_first"]
        )
        self._action_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["action_space"]
        self._reward_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["reward_space"](self._cfg["frame_skip"])

        self._save_replay_gif = cfg.save_replay_gif
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_count = 0

    def reset(self) -> np.ndarray:
        if not self._init_flag:

            self._env = dmc2gym.make(
                domain_name=self._cfg["domain_name"],
                task_name=self._cfg["task_name"],
                seed=1,
                visualize_reward=self._cfg["visualize_reward"],
                from_pixels=self._cfg["from_pixels"],
                height=self._cfg["height"],
                width=self._cfg["width"],
                frame_skip=self._cfg["frame_skip"],
                channels_first=self._cfg["channels_first"],
                render_image=self._cfg["render_image"]
            )

            # optional env wrapper
            if self._cfg['warp_frame']:
                self._env = WarpFrameWrapper(self._env, size=self._cfg['resize'])
            if self._cfg['scale']:
                self._env = ScaledFloatFrameWrapper(self._env)
            if self._cfg['clip_rewards']:
                self._env = ClipRewardWrapper(self._env)
            if self._cfg['action_repeat']:
                self._env = ActionRepeatWrapper(self._env, self._cfg['action_repeat'])
            if self._cfg['frame_stack'] > 1:
                self._env = FrameStackWrapper(self._env, self._cfg['frame_stack'])

            # set the obs, action space of wrapped env
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space

            if self._replay_path is not None:
                if gym.version.VERSION > '0.22.0':
                    self._env.metadata.update({'render_modes': ["rgb_array"]})
                else:
                    self._env.metadata.update({'render.modes': ["rgb_array"]})
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )
                self._env.start_video_recorder()

            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)

        self._eval_episode_return = 0
        obs = self._env.reset()

        obs = obs['state']
        obs = to_ndarray(obs).astype(np.float32)

        self._current_step = 0
        if self._save_replay_gif:
            self._frames = []

        # 新增：用于存储每一局的动作和奖励
        self._episode_actions = []
        self._episode_rewards = []

        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        action = action.astype('float32')
        action = affine_transform(action, min_val=self._env.action_space.low, max_val=self._env.action_space.high)
        obs, rew, done, info = self._env.step(action)
        self._current_step += 1

        # print(f'action: {action}, obs: {obs}, rew: {rew}, done: {done}, info: {info}')
        print(f'step {self._current_step}: action: {action}, rew: {rew}, done: {done}')

        self._eval_episode_return += rew

        # 记录动作和奖励
        self._episode_actions.append(action)
        self._episode_rewards.append(rew)

        if self._cfg["from_pixels"]:
            obs = obs
        else:
            info['image_obs'] = info['image_obs'].copy()
            image_obs = info['image_obs']

        if self._save_replay_gif:
            self._frames.append(image_obs)

        if done:
            info['eval_episode_return'] = self._eval_episode_return

            if not os.path.exists(self._replay_path_gif):
                os.makedirs(self._replay_path_gif)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            path = os.path.join(
                self._replay_path_gif,
                '{}_episode_{}_seed{}_{}.gif'.format(f'{self._cfg["domain_name"]}_{self._cfg["task_name"]}',
                                                     self._save_replay_count, self._seed, timestamp)
            )
            self.display_frames_as_gif(self._frames, path)
            print(f'save episode {self._save_replay_count} in {self._replay_path_gif}!')
            self._save_replay_count += 1

            # 绘制并保存动作分布和奖励变化图
            self.plot_action_distribution_and_rewards()

            # 清空记录以便下一局使用
            self._episode_actions = []
            self._episode_rewards = []

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)  # wrapped to be transferred to a array with shape (1,)


        return BaseEnvTimestep(obs, rew, done, info)

    def plot_action_distribution_and_rewards(self):
        # 将动作转换为NumPy数组以便更好地进行处理
        actions = np.array(self._episode_actions)
        rewards = np.array(self._episode_rewards)

        # 绘制动作分布图
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        for action_dim in range(actions.shape[1]):
            axs[0].hist(actions[:, action_dim], bins=50, alpha=0.5, label=f'Action Dimension {action_dim+1}')

        axs[0].set_title('Action Distribution')
        axs[0].set_xlabel('Action Value')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()

        # 绘制奖励变化图
        axs[1].plot(rewards, label="Reward", color='blue')
        axs[1].set_title('Reward Over Time')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Reward')
        axs[1].legend()

        # 保存图表
        if not os.path.exists(self._replay_path_gif):
            os.makedirs(self._replay_path_gif)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plot_path = os.path.join(self._replay_path_gif, f'episode_{self._save_replay_count}_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close(fig)

        print(f'Action distribution and reward plot saved at {plot_path}')

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        # 调整每一帧的维度
        # frames = [np.transpose(frame, (1, 2, 0)) for frame in frames]

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='pillow', fps=20)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample().astype(np.float32)
        return random_action

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
        return "DI-engine DeepMind Control Suite to gym Env: " + self._cfg["domain_name"] + ":" + self._cfg["task_name"]
