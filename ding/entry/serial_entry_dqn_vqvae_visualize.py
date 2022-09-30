from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.utils import set_pkg_seed
from .utils import random_collect
import copy


def serial_pipeline_dqn_vqvae_visualize(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(3e3),
        obs = None,
name_suffix=None,
visualize_path =None,
        number_of_frames=None,
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)

    # save replay
    # evaluator_env.enable_save_replay(replay_path=cfg.env.replay_path)

    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # load pretrained model
    if cfg.policy.model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location='cpu'))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    cfg.policy.other.replay_buffer.replay_buffer_size = cfg.policy.replay_buffer_size_vqvae
    replay_buffer_vqvae = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)

    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )

    # visualize: analyze vqvae encoder
    # policy.visualize_latent(save_histogram=False, save_mapping=True, name_suffix='lunarlander_k8_seed1_best', granularity=0.01)
    # policy.visualize_latent(save_histogram=True, save_mapping=False, name_suffix='lunarlander_k8_seed1_best', granularity=0.01)

    # policy.visualize_latent(save_histogram=False, save_mapping=True, name_suffix='lunarlander_k8_seed1_iter2e5', granularity=0.01)
    # policy.visualize_latent(save_histogram=True, save_mapping=False, name_suffix='lunarlander_k8_seed1_iter2e5', granularity=0.01)

    # visualize: analyze vqvae
    if number_of_frames is not None:
        # process <number_of_frames> frames once
        for timestep in range(number_of_frames):
            name_suffix_timestep = copy.deepcopy(name_suffix)
            name_suffix_timestep = name_suffix_timestep + f'_t{timestep}'
            policy.visualize_latent(save_histogram=False, save_mapping=False, save_decoding_mapping=True,
                                    obs=obs[timestep],
                                    name_suffix=name_suffix_timestep, granularity=0.01, k=8,
                                    visualize_path=visualize_path)
            # policy.visualize_latent(save_histogram=False, save_mapping=True, save_decoding_mapping=False, obs=obs[timestep],
            #                         name_suffix=name_suffix_timestep, granularity=0.01, k=8, visualize_path=visualize_path)
            from ding.torch_utils import Adam, to_device, to_tensor
            # if cfg.policy.cuda:
            #     obs = to_device(obs, 'cuda')
            # policy._eval_model.eval()
            # with torch.no_grad():
            #     output = policy._eval_model.forward(obs[timestep])
            # print('action:', output['action'])
    else:
        # process one frame once
        policy.visualize_latent(save_histogram=False, save_mapping=False, save_decoding_mapping=True, obs=obs, name_suffix= name_suffix, granularity=0.01, k=8, visualize_path =visualize_path )
        # policy.visualize_latent(save_histogram=False, save_mapping=True, save_decoding_mapping=False, obs=obs, name_suffix= name_suffix, granularity=0.01, k=8, visualize_path =visualize_path )

        # for moving
        # policy.visualize_latent(save_histogram=False, save_mapping=True, save_decoding_mapping=False, obs=obs, name_suffix= name_suffix, granularity=0.01, k=16, visualize_path =visualize_path )


        from ding.torch_utils import Adam, to_device, to_tensor
        # if cfg.policy.cuda:
        #     obs = to_device(obs, 'cuda')

        # policy._eval_model.eval()
        # with torch.no_grad():
        #     output = policy._eval_model.forward(obs)
        # print('action:', output['action'])
