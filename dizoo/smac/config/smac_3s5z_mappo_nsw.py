from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

agent_num = 8
# collector_env_num = 8
# evaluator_env_num = 8
collector_env_num = 1  # TODO(pu)
evaluator_env_num = 1
special_global_state = True

main_config = dict(
    exp_name='smac_3s5z_ppo',
    env=dict(
        map_name='3s5z',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=16,  # TODO(pu)
        # n_evaluator_episode=1,
        stop_value=0.99,
        death_mask=False,
        special_global_state=special_global_state,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        action_space='discrete',
        share_weight=False,  # TODO(pu)
        agent_num=agent_num,
        model=dict(
            # (int) agent_num: The number of the agent.
            # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
            agent_num=agent_num,
            # (int) obs_shape: The shapeension of observation of each agent.
            # For 3s5z, obs_shape=150; for 2c_vs_64zg, agent_num=404.
            # (int) global_obs_shape: The shapeension of global observation.
            # For 3s5z, obs_shape=295; for 2c_vs_64zg, agent_num=342.
            agent_obs_shape=150,
            global_obs_shape=295,
            # (int) action_shape: The number of action which each agent can take.
            # action_shape= the number of common action (6) + the number of enemies.
            # For 3s5z, obs_shape=14 (6+8); for 2c_vs_64zg, agent_num=70 (6+64).
            action_shape=14,
            # delete encode in code
            actor_head_hidden_size=256,
            critic_head_hidden_size=512,
        ),
        # used in state_num of hidden_state
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            epoch_per_collect=5,
            batch_size=1600,
            learning_rate=5e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.001,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
            ignore_done=False,
        ),
        collect=dict(env_num=collector_env_num, n_sample=6400),
        eval=dict(
            evaluator=dict(
                eval_freq=200,
            ),
            env_num=evaluator_env_num,
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)


def train(args):
    main_config.exp_name='debug_smac_3s5z_mappo_nsw'+'_seed'+f'{args.seed}'

    import copy
    # 3125 iterations= 10M/3200 env steps mmm2
    # 6250 iterations= 10M/1600 env steps 5m6m
    # 1562.5 iterations= 5M/3200 env steps mmm
    # 3125 iterations= 5M/1600 env steps 3s5z
    # serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed,  max_iterations=3125)

    # set max total env steps in serial_entry
    serial_pipeline_onpolicy([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed)


if __name__ == "__main__":
    import argparse
    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)