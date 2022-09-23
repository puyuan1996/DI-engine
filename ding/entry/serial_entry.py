from typing import Union, Optional, List, Any, Tuple
import os
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector, create_serial_evaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from .utils import random_collect

import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np


def serial_pipeline(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for off-policy RL.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    collect_cfg = copy.deepcopy(input_cfg)
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
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
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # load model
    # policy.collect_mode.load_state_dict(torch.load(cfg.policy.model_path, map_location='cuda'))

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
    evaluator = create_serial_evaluator(
        cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            """
            ### plot action histogram
            from ding.entry import collect_episodic_demo_data
            collect_episodic_demo_data(
                copy.deepcopy(collect_cfg),
                collect_count=1,
                seed=cfg.seed,
                expert_data_path=f'/home/puyuan//DI-engine/hopper_sac_action_bins_seed{cfg.seed}/expert_data_1eps_iter{learner.train_iter}.pkl',
                state_dict=torch.load(f'/home/puyuan//DI-engine/hopper_sac_action_bins_seed{cfg.seed}/ckpt/ckpt_best.pth.tar', map_location='cpu')
            )
            with open(f'/home/puyuan//DI-engine/hopper_sac_action_bins_seed{cfg.seed}/expert_data_1eps_iter{learner.train_iter}.pkl', 'rb') as f:
                data = pickle.load(f)
            episode_actions = torch.stack([data[0][i]['action'] for i in range(len(data[0]))],axis=0)
            
            for action_dim in range(3):
                fig = plt.figure()
                # Fixing bin edges
                HIST_BINS = np.linspace(-1, 1, 20)
                # the histogram of the data
                n, bins, patches = plt.hist(
                    episode_actions[:,action_dim].cpu().numpy(), HIST_BINS, density=False, facecolor='g', alpha=0.75
                )
                plt.xlabel(f'actions dim {action_dim}')
                plt.ylabel('Count')
                plt.title(f'Histogram of actions dim {action_dim}')
                plt.grid(True)
                plt.show()
                # plt.savefig(f'/home/puyuan//DI-engine/hopper_sac_action_bins_seed{cfg.seed}/hopper-v3_sac_episode_actions_{action_dim}dim_histogram_iter{learner.train_iter}.png')
                tb_logger.add_histogram(f'hopper-v3_sac_episode_actions_{action_dim}dim_histogram_{cfg.seed}', episode_actions[:,action_dim], learner.train_iter)
            """
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        


        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    # import time
    # import pickle
    # import numpy as np
    # with open(os.path.join(cfg.exp_name, 'result.pkl'), 'wb') as f:
    #     eval_value_raw = [d['final_eval_reward'] for d in eval_info]
    #     final_data = {
    #         'stop': stop,
    #         'env_step': collector.envstep,
    #         'train_iter': learner.train_iter,
    #         'eval_value': np.mean(eval_value_raw),
    #         'eval_value_raw': eval_value_raw,
    #         'finish_time': time.ctime(),
    #     }
    #     pickle.dump(final_data, f)
    return policy
