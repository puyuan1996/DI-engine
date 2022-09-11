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


def serial_pipeline_dqn_vqvae(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(3e3),
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

    # visualize: analyze vqvae decoder
    # policy.visualize_latent(save_histogram=False, save_mapping=False, save_decoding_mapping=True, name_suffix='lunarlander_obs0_k8_seed1_neg1_best', granularity=0.01,k=8)
    # policy.visualize_latent(save_histogram=False, save_mapping=False, save_decoding_mapping=True, name_suffix='lunarlander_k8_seed1_iter2e5', granularity=0.01)

    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        # random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)
        if cfg.policy.get('transition_with_policy_data', False):
            collector.reset_policy(policy.collect_mode)
        else:
            action_space = collector_env.action_space
            random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
            collector.reset_policy(random_policy)

        collect_kwargs = commander.step()
        new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
        for item in new_data:
            item['warm_up'] = True
        replay_buffer.push(new_data, cur_collector_envstep=0)
        collector.reset_policy(policy.collect_mode)

        # ====================
        # warm_up phase: train VAE
        # ====================
        # Learn policy from collected data
        for i in range(cfg.policy.warm_up_update):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = replay_buffer.sample(cfg.policy.learn.vqvae_batch_size, learner.train_iter)
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
            if learner.policy.get_attribute('warm_up_stop'):
                break
        replay_buffer.clear()  # TODO(pu): NOTE

    # NOTE: for the case collector_env_num>1, because after the random collect phase,  self._traj_buffer[env_id] may be not empty. Only
    # if the condition "timestep.done or len(self._traj_buffer[env_id]) == self._traj_len" is satisfied, the self._traj_buffer will be clear.
    # For our alg., the data in self._traj_buffer[env_id], latent_action=False, cannot be used in rl_vae phase.
    collector.reset(policy.collect_mode)

    for iter in range(max_iterations):
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        for item in new_data:
            item['warm_up'] = False
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        replay_buffer_vqvae.push(copy.deepcopy(new_data), cur_collector_envstep=collector.envstep)

        # ====================
        # RL phase
        # ====================
        if iter % cfg.policy.learn.rl_vae_update_circle in range(0, cfg.policy.learn.rl_vae_update_circle):
            # Learn policy from collected data
            for i in range(cfg.policy.learn.update_per_collect_rl):
                # Learner will train ``update_per_collect`` times in one iteration.
                train_data = replay_buffer.sample(cfg.policy.learn.rl_batch_size, learner.train_iter)
                if train_data is not None:
                    for item in train_data:
                        item['rl_phase'] = True
                        item['vae_phase'] = False
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
        # ====================
        # VAE phase
        # ====================
        if not cfg.policy.vqvae_pretrain_only:
            if iter % cfg.policy.learn.rl_vae_update_circle in range(cfg.policy.learn.rl_vae_update_circle - 1,
                                                                     cfg.policy.learn.rl_vae_update_circle):
                for i in range(cfg.policy.learn.update_per_collect_vae):
                    train_data_vqvae = replay_buffer_vqvae.sample(
                        int(cfg.policy.learn.vqvae_batch_size), learner.train_iter
                    )
                    train_data = train_data_vqvae

                    if train_data is not None:
                        for item in train_data:
                            item['rl_phase'] = False
                            item['vae_phase'] = True
                    if train_data is None:
                        # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                        logging.warning(
                            "Replay buffer's data can only train for {} steps. ".format(i) +
                            "You can modify data collect config, e.g. increasing n_sample, n_episode."
                        )
                        break
                    learner.train(train_data, collector.envstep)
                    if learner.policy.get_attribute('priority_vqvae'):
                        replay_buffer_vqvae.update(learner.priority_info)

        if collector.envstep > max_env_step:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy
