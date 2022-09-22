from easydict import EasyDict
num_actuators=10
from itertools import product
action_mask = list(product(*[list(range(2)) for dim in range(num_actuators)] ))

gym_hybrid_mpdqn_config = dict(
    # exp_name='gym_hybrid_mpdqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        # (bool) Scale output action into legal range [-1, 1].
        act_scale=True,
        # env_id='Sliding-v0',
        # env_id='Moving-v0',
        env_id='HardMove-v0',
        num_actuators=num_actuators,  # only for 'HardMove-v0'
        n_evaluator_episode=8,
        # stop_value=2,
        stop_value=999,
        save_replay_gif=False,
        replay_path=None,
    ),
    policy=dict(
        model_path=None,
        # cuda=True,
        cuda=False,

        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        discount_factor=0.99,
        nstep=1,
        model=dict(
            obs_shape=10,
            # for 'Sliding-v0','Moving-v0'
            # action_shape=dict(
            #     action_type_shape=3,
            #     action_args_shape=2,
            # ),
            # action_mask=[[1, 0], [0, 1], [0, 0]],
            # for 'HardMove-v0'
            action_shape=dict( 
                    action_type_shape=int(2** num_actuators),
                    action_args_shape=int(num_actuators),
                ),
            action_mask=action_mask,
            multi_pass=True,
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=500,  # 10~500
            batch_size=320,
            learning_rate_dis=3e-4,
            learning_rate_cont=3e-4,
            target_theta=0.001,
            update_circle=10,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            n_sample=3200,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=1,
                end=0.1,
                # (int) Decay length(env step)
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ),
        ),
    )
)

gym_hybrid_mpdqn_config = EasyDict(gym_hybrid_mpdqn_config)
main_config = gym_hybrid_mpdqn_config

gym_hybrid_mpdqn_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='pdqn'),
)
gym_hybrid_mpdqn_create_config = EasyDict(gym_hybrid_mpdqn_create_config)
create_config = gym_hybrid_mpdqn_create_config

# if __name__ == "__main__":
#     # or you can enter `ding -m serial -c gym_hybrid_mpdqn_config.py -s 0`
#     from ding.entry import serial_pipeline
#     serial_pipeline([main_config, create_config], seed=0)

def train(args):
    main_config.exp_name = 'data_hardmove_n10/mpdqn' + '_seed' + f'{args.seed}'+'_3M'

    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed,max_env_step=int(3e6))


if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline

    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)