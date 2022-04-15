from typing import Union, Optional, List, Any, Callable, Tuple
import pickle
import torch
from functools import partial
from ding.config import compile_config, read_config
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.torch_utils import to_tensor, to_ndarray, tensor_to_list
import os


def eval(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        eval_episodes_num=5,
        model: Optional[torch.nn.Module] = None,
        model_path = None,
        state_dict: Optional[dict] = None,
) -> float:
    r"""
    Overview:
        Pure evaluation entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type += '_command'
    cfg = compile_config(cfg, auto=True, create_cfg=create_cfg)
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    env = env_fn(evaluator_env_cfg[0])
    env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['eval']).eval_mode
    if state_dict is None:
        state_dict = torch.load(model_path, map_location='cpu')
    policy.load_state_dict(state_dict)
    # NOTE
    env.enable_save_replay(replay_path=replay_path)
    for i in range(eval_episodes_num):
        eps_len=0
        obs = env.reset()
        obs = {0: obs}
        eval_episode_return = 0.
        while True:
            eps_len+=1
            policy_output = policy.forward(obs)

            actions = {i: a['action'] for i, a in policy_output.items()}
            actions = to_ndarray(actions)

            action = policy_output[0]['action']
            action = to_ndarray(action)
            timestep = env.step(action)
            # print(action)
            # print(timestep.reward)

            timesteps = {0: timestep}
            timesteps = to_tensor(timesteps, dtype=torch.float32)

            # print(timestep.info)
            eval_episode_return += timestep.reward

            obs = timestep.obs
            obs = {0: obs}

            if timestep.done:
                print(timestep.info)
                break
        print(f'Episode {i} done! The episode length is {eps_len}. The episode return is {eval_episode_return}. The last reward is {timestep.reward}.')

if __name__ == "__main__":
    module_path = os.path.dirname(__file__)

    # NOTE: keeper mappo
    # model_path = '/home/puyuan/DI-engine/gfootball_keeper_mappo_seed3/ckpt/ckpt_best.pth.tar'
    # cfg = '/home/puyuan/DI-engine/dizoo/gfootball/config/gfootball_keeper_mappo_config.py'
    # replay_path = module_path + '/keeper_mappo_video'

    # NOTE: keeper masac
    model_path = '/home/puyuan/DI-engine/gfootball_keeper_masac_seed0/ckpt/ckpt_best.pth.tar'
    cfg = '/home/puyuan/DI-engine/dizoo/gfootball/config/gfootball_keeper_masac_config.py'
    replay_path = module_path + '/keeper_masac_video'

    # for i in range(3,4):
    eval(cfg, seed=0, eval_episodes_num=10,  model_path=model_path)