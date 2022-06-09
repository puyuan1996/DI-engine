from easydict import EasyDict

nstep = 3
each_dim_disc_size=7  # n: discrete size of each dim in origin continuous action

hopper_dqn_default_config = dict(
    exp_name='hopper_dqn_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=int(1e6),  # stop according to max env steps 
        each_dim_disc_size=each_dim_disc_size,  # n: discrete size of each dim in origin continuous action
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        priority=False,
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        model=dict(
            obs_shape=11,
            # NOTE：original_action_shape m=3, 
            # action_shape=int(64),  # num of num_embeddings: K = n**m e.g. 4**3=64

            action_shape=each_dim_disc_size,  # n
            agent_num=3,  # m
            
            # encoder_hidden_size_list=[128, 128, 64],  # small net
            encoder_hidden_size_list=[256, 256, 128],  # middle net
            # encoder_hidden_size_list=[512, 512, 256],  # large net
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            ignore_done=False,
            batch_size=512,
            learning_rate=3e-4,
            # Frequency of target network update.
            target_update_freq=100,
            update_per_collect=20,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=256,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=1,
                end=0.05,
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), )
        ),
    ),
)

hopper_dqn_default_config = EasyDict(hopper_dqn_default_config)
main_config = hopper_dqn_default_config

hopper_dqn_create_config = dict(
    env=dict(
        type='mujoco-disc',
        import_names=['dizoo.mujoco.envs.mujoco_env_disc'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn_ma'),
)
hopper_dqn_create_config = EasyDict(hopper_dqn_create_config)
create_config = hopper_dqn_create_config


def train(args):
    main_config.exp_name = 'data_hopper/dqn_ma_n7m3_middlenet_upc20' + '_seed' + f'{args.seed}'+'_3M'
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