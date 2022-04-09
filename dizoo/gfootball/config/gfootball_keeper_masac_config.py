from easydict import EasyDict

agent_num = 3
collector_env_num = 1
evaluator_env_num = 1
# special_global_state = True
# masac 5m6m config -> keeper
gfootball_keeper_masac_default_config = dict(
    exp_name='gfootball_keeper_masac_seed0',
    env=dict(
        # map_name='academy_3_vs_1_with_keeper',
        env_name='academy_3_vs_1_with_keeper',
        difficulty=7,
        # reward_only_positive=True,
        # mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=32,
        stop_value=0.99,
        # death_mask=True,
        # special_global_state=special_global_state,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        random_collect_size=0,
        # random_collect_size=int(1e4),
        model=dict(
            agent_obs_shape=26,
            global_obs_shape=52,
            action_shape=19,
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            learning_rate_alpha=5e-5,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=True,
            log_space=True,
        ),
        collect=dict(
            env_num=collector_env_num,
            n_sample=1600,
            unroll_len=1,
        ),
        eval=dict(
            evaluator=dict(eval_freq=50, ),
            env_num=evaluator_env_num,
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=int(5e4),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ), ),
    ),
)

gfootball_keeper_masac_default_config = EasyDict(gfootball_keeper_masac_default_config)
main_config = gfootball_keeper_masac_default_config

gfootball_keeper_masac_default_create_config = dict(
    env=dict(
        type='keeper',
        # import_names=['dizoo.gfootball.envs.gfootball_env'],
        import_names=['dizoo.gfootball.envs.academy_3_vs_1_with_keeper'],

    ),
    env_manager=dict(type='base'),
    policy=dict(type='sac_discrete', ),
)
gfootball_keeper_masac_default_create_config = EasyDict(gfootball_keeper_masac_default_create_config)
create_config = gfootball_keeper_masac_default_create_config


# if __name__ == "__main__":
#     serial_pipeline([main_config, create_config], seed=0)

def train(args):
    from ding.entry import serial_pipeline
    main_config.exp_name='debug_gfootball_keeper_masac_'+'seed'+f'{args.seed}'+'_rcs0'
    import copy
    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=4e6)


if __name__ == "__main__":
    import argparse
    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        
        train(args)
