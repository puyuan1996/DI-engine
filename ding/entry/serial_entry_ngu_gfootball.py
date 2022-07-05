from typing import Union, Optional, List, Any, Tuple
import os
import torch
from ditk import logging
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.reward_model import create_reward_model
from ding.utils import set_pkg_seed
from .utils import random_collect
import copy
from ding.torch_utils import to_device


def serial_pipeline_ngu_gfootball(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry for NGU. The corresponding paper is
        `never give up: learning directed exploration strategies`.
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
    # if you want to save replay, please uncomment this line
    # evaluator_env.enable_save_replay(cfg.env.replay_path)

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

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
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )

    # for gfootball
    rnd_reward_model = create_reward_model(cfg.rnd_reward_model, policy.collect_mode.get_attribute('device'), tb_logger)
    episodic_reward_model = create_reward_model(
        cfg.episodic_reward_model, policy.collect_mode.get_attribute('device'), tb_logger
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)

    estimate_cnt = 0
    iter_ = 0
    while True:
        """some hyper-parameters used in NGU"""
        # index_to_eps = {i: 0.4 ** (1 + 8 * i / (self._env_num - 1)) for i in range(self._env_num)}
        # index_to_beta = {
        #     i: 0.3 * torch.sigmoid(torch.tensor(10 * (2 * i - (collector_env_num - 2)) / (collector_env_num - 2)))
        #     for i in range(collector_env_num)
        # }
        # index_to_gamma = {
        #     i: 1 - torch.exp(
        #         (
        #             (collector_env_num - 1 - i) * torch.log(torch.tensor(1 - 0.997)) +
        #             i * torch.log(torch.tensor(1 - 0.99))
        #         ) / (collector_env_num - 1)
        #     )
        #     for i in range(collector_env_num)
        # }
        iter_ += 1

        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=None)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)

        # # collect data for reward_model training
        # rnd_reward_model.collect_data(new_data)
        # episodic_reward_model.collect_data(new_data)

        # for gfootball
        # obs in data_reward_model is encoding from trained encoder.
        # for rnd reward model
        new_data_reward_model_rnd = copy.deepcopy(new_data)
        if cfg.policy.cuda:
            new_data_reward_model_rnd = to_device(new_data_reward_model_rnd, policy.collect_mode.get_attribute('device'))
        batch_size = len(new_data_reward_model_rnd)
        timesteps = len(new_data_reward_model_rnd[0]['obs']['processed_obs'])
        for bs in range(batch_size):
            obs_tmp = []
            for timestep in range(timesteps):
                obs_tmp.append(rnd_reward_model.encoder(new_data_reward_model_rnd[bs]['obs']['processed_obs'][timestep]).view(-1))
            new_data_reward_model_rnd[bs]['obs'] = obs_tmp

        # for episodic reward model
        new_data_reward_model_episodic = copy.deepcopy(new_data)
        if cfg.policy.cuda:
            new_data_reward_model_episodic = to_device(new_data_reward_model_episodic, policy.collect_mode.get_attribute('device'))
        batch_size = len(new_data_reward_model_episodic)
        timesteps = len(new_data_reward_model_episodic[0]['obs']['processed_obs'])
        for bs in range(batch_size):
            obs_tmp = []
            for timestep in range(timesteps):
                obs_tmp.append(episodic_reward_model.encoder(new_data_reward_model_episodic[bs]['obs']['processed_obs'][timestep]).view(-1))
            new_data_reward_model_episodic[bs]['obs'] = obs_tmp

        # collect data for reward_model training
        rnd_reward_model.collect_data(new_data_reward_model_rnd)
        episodic_reward_model.collect_data(new_data_reward_model_episodic)

        # update reward_model
        rnd_reward_model.train()
        if (iter_ + 1) % cfg.rnd_reward_model.clear_buffer_per_iters == 0:
            rnd_reward_model.clear_data()
        episodic_reward_model.train()
        if (iter_ + 1) % cfg.episodic_reward_model.clear_buffer_per_iters == 0:
            episodic_reward_model.clear_data()

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
            
            # use the newest reward model to estimate intrinsic reward
            with torch.no_grad():
                # for rnd reward model
                train_data_reward_model_rnd = copy.deepcopy(train_data)
                if cfg.policy.cuda:
                    train_data_reward_model_rnd = to_device(train_data_reward_model_rnd, policy.collect_mode.get_attribute('device'))
                batch_size = len(train_data_reward_model_rnd)
                timesteps = len(train_data_reward_model_rnd[0]['obs']['processed_obs'])
                for bs in range(batch_size):
                    obs_tmp = []
                    for timestep in range(timesteps):
                        obs_tmp.append(rnd_reward_model.encoder(train_data_reward_model_rnd[bs]['obs']['processed_obs'][timestep]).view(-1))
                    train_data_reward_model_rnd[bs]['obs'] = obs_tmp

                # for episodic reward model
                train_data_reward_model_episodic = copy.deepcopy(train_data)
                if cfg.policy.cuda:
                    train_data_reward_model_episodic = to_device(train_data_reward_model_episodic, policy.collect_mode.get_attribute('device'))
                batch_size = len(train_data_reward_model_episodic)
                timesteps = len(train_data_reward_model_episodic[0]['obs']['processed_obs'])
                for bs in range(batch_size):
                    obs_tmp = []
                    for timestep in range(timesteps):
                        obs_tmp.append(episodic_reward_model.encoder(train_data_reward_model_episodic[bs]['obs']['processed_obs'][timestep]).view(-1))
                    train_data_reward_model_episodic[bs]['obs'] = obs_tmp

            # calculate the inter-episodic and episodic intrinsic reward
            rnd_reward = rnd_reward_model.estimate(train_data_reward_model_rnd)
            episodic_reward = episodic_reward_model.estimate(train_data_reward_model_episodic)

            # update train_data reward using the augmented reward
            train_data_augmented, estimate_cnt = episodic_reward_model.fusion_reward(
                train_data,
                rnd_reward,
                episodic_reward,
                nstep=cfg.policy.nstep,
                collector_env_num=cfg.policy.collect.env_num,
                tb_logger=tb_logger,
                estimate_cnt=estimate_cnt
            )
            learner.train(train_data_augmented, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
