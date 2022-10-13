from easydict import EasyDict

agent_num = 4
obs_dim = 34

collector_env_num = 4
evaluator_env_num = 8

main_config = dict(
    exp_name='gfootball_counter_madqn_seed0',
    env=dict(
        env_name='academy_counterattack_hard',
        agent_num=agent_num,
        obs_dim=obs_dim,
        n_evaluator_episode=32,
        stop_value=1,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        # share_weight=True,
        # multi_agent=True,
        model=dict(
            agent_num=agent_num,
            obs_shape=obs_dim,
            global_obs_shape=int(obs_dim * 2),
            action_shape=19,
            global_boost=True,
            hidden_size_list=[256, 256],
            mixer=False,
            lstm_type='gru',
            dueling=False,
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # ==============================================================
            # The following configs is algorithm-specific
            update_per_collect=20,
            batch_size=64,
            learning_rate=0.0005,
            clip_value=5,
            double_q=False,
            iql=False,
            target_update_theta=0.008,
            discount_factor=0.95,
        ),
        collect=dict(
            n_episode=32,
            unroll_len=10,
            env_num=collector_env_num,),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=10000,
            ),
            replay_buffer=dict(
                replay_buffer_size=15000,
                # (int) The maximum reuse times of each data
                max_reuse=1e+9,
                max_staleness=1e+9,
            ),
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='gfootball-academy',
        import_names=['dizoo.gfootball.envs.gfootball_academy_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='madqn'),
    collector=dict(type='episode', get_train_sample=True),
)
create_config = EasyDict(create_config)


# if __name__ == "__main__":
#     # or you can enter `ding -m serial_onpolicy -c gfootball_counter_madqn_config.py -s 0`
#     from ding.entry import serial_pipeline_onpolicy
#     serial_pipeline_onpolicy([main_config, create_config], seed=0)

def train(args):
    main_config.exp_name = 'data_counter/madqn' + '_seed' + f'{args.seed}' + '_10M'
    serial_pipeline_onpolicy(
        [copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(10e6)
    )


if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline_onpolicy

    for seed in [0, 1, 2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        train(args)
