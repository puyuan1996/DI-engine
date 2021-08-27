from typing import Dict
import gym
import numpy as np

from ding.envs import ObsNormEnv, RewardNormEnv


def wrap_mujoco(env_id, norm_obs: Dict=dict(use_norm=False, ), 
                norm_reward: Dict=dict(use_norm=False, ), only_info=False) -> gym.Env:
    r"""
    Overview:
        Wrap Mujoco Env to preprocess env step's return info, e.g. observation normalization, reward normalization, etc.
    Arguments:
        - env_id (:obj:`str`): Mujoco environment id, for example "HalfCheetah-v3"
        - norm_obs (:obj:`EasyDict`): Whether to normalize observation or not
        - norm_reward (:obj:`EasyDict`): Whether to normalize reward or not. For evaluator, environment's reward \
            should not be normalized: Either ``norm_reward`` is None or ``norm_reward.use_norm`` is False can do this.
    Returns:
        - wrapped_env (:obj:`gym.Env`): The wrapped mujoco environment
    """
    if not only_info:
        env = gym.make(env_id)
        if norm_obs is not None and norm_obs.use_norm:
            env = ObsNormEnv(env)
        if norm_reward is not None and norm_reward.use_norm:
            env = RewardNormEnv(env, norm_reward.reward_discount)
        return env
    else:
        wrapper_info = ''
        if norm_obs is not None and norm_obs.use_norm:
            wrapper_info = ObsNormEnv.__name__ + '\n'
        if norm_reward is not None and norm_reward.use_norm:
            wrapper_info += RewardNormEnv.__name__ + '\n'
        return wrapper_info
