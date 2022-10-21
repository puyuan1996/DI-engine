from easydict import EasyDict

agent_num = 3
obs_dim = 26

collector_env_num = 8
evaluator_env_num = 8

main_config = dict(
    exp_name='gfootball_keeper_madqn_seed0',
    env=dict(
        env_name='academy_3_vs_1_with_keeper',
        agent_num=agent_num,
        obs_dim=obs_dim,
        n_evaluator_episode=8,
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
        # multi_agent=True,
        # nstep=3,
        model=dict(
            agent_num=agent_num,
            obs_shape=obs_dim,
            global_obs_shape=int(obs_dim * 2),
            action_shape=19,
            global_boost=True,
            hidden_size_list=[256, 256],
            # hidden_size_list=[256, 128],
            mixer=False,
            lstm_type='gru',
            dueling=False,
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # ==============================================================
            # The following configs is algorithm-specific
            update_per_collect=50,
            batch_size=64,
            # batch_size=320,
            learning_rate=0.0005,
            clip_value=10,
            double_q=False,
            iql=False,
            target_update_theta=0.005,
            discount_factor=0.99,
        ),
        collect=dict(
            n_episode=16,  # episode_length=100
            unroll_len=10,
            # unroll_len=50,
            env_num=collector_env_num,),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=50000,
            ),
            replay_buffer=dict(
                # replay_buffer_size=int(5e5),
                replay_buffer_size=int(1e5),
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
    # policy=dict(type='madqn_nstep'),
    policy=dict(type='madqn'),
    collector=dict(type='episode', get_train_sample=True),
)

create_config = EasyDict(create_config)

# if __name__ == "__main__":
#     # or you can enter `ding -m serial_onpolicy -c gfootball_keeper_madqn_config.py -s 0`
#     from ding.entry import serial_pipeline_onpolicy
#     serial_pipeline_onpolicy([main_config, create_config], seed=0)

def train(args):
    main_config.exp_name = 'data_kepper/madqn_ul10' + '_seed' + f'{args.seed}' + '_4M'
    serial_pipeline(
        [copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(4e6)
    )


if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline

    for seed in [0, 1, 2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        train(args)