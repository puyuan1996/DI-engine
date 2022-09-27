from easydict import EasyDict

nstep = 3
lunarlander_dqn_config = dict(
    exp_name='lunarlander_dqn_seed0',
    env=dict(
        env_id='LunarLanderContinuous-v2',
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # Env number respectively for collector and evaluator.
        act_scale=True,
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        # stop_value=200,
        stop_value=int(1e6),  # stop according to max env steps 
        # The path to save the game replay
        replay_path='./lunarlander_dqn_seed0/video',
        save_replay_gif=False,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        # load_path="./lunarlander_dqn_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=8,
            action_shape=8,
            # encoder_hidden_size_list=[512, 64],
            encoder_hidden_size_list=[128, 128, 64],  # small net
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            update_per_collect=50,
            batch_size=512,
            learning_rate=3e-4,
            # Frequency of target network update.
            target_update_freq=500,
            target_update_theta=0.001,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=256,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                # type='exp',
                # start=0.95,
                # end=0.1,
                # decay=50000,
                type='exp',
                start=1,
                end=0.05,
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), )
            # replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
lunarlander_dqn_config = EasyDict(lunarlander_dqn_config)
main_config = lunarlander_dqn_config

lunarlander_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    # env_manager=dict(type='base'),
    policy=dict(type='dqn_cluster'),
)
lunarlander_dqn_create_config = EasyDict(lunarlander_dqn_create_config)
create_config = lunarlander_dqn_create_config

# if __name__ == "__main__":
#     # or you can enter `ding -m serial -c lunarlander_dqn_config.py -s 0`
#     from ding.entry import serial_pipeline
#     serial_pipeline([main_config, create_config], seed=0)

def train(args):
    main_config.exp_name = 'data_lunarlander/dqn_cluster_k8_' + '_seed' + f'{args.seed}'
    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(3e6))

if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline
    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        train(args)
        # obs_shape: 8
        # action_shape: 2