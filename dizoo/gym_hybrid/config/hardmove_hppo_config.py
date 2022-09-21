from easydict import EasyDict
num_actuators=10
from itertools import product
action_mask = list(product(*[list(range(2)) for dim in range(num_actuators)] ))

gym_hybrid_hppo_config = dict(
    exp_name='gym_hybrid_hppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        # (bool) Scale output action into legal range, usually [-1, 1].
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
        cuda=True,
        priority=False,
        action_space='hybrid',
        recompute_adv=True,
        model=dict(
            obs_shape=10,
            # for 'Sliding-v0','Moving-v0'
            # action_shape=dict(
            #     action_type_shape=3,
            #     action_args_shape=2,
            # ),
            # for 'HardMove-v0'
            action_shape=dict( 
                    action_type_shape=int(2** num_actuators),
                    action_args_shape=int(num_actuators),
                ),
            action_space='hybrid',
            encoder_hidden_size_list=[256, 128, 64, 64],
            sigma_type='fixed',
            fixed_sigma_value=0.3,
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.03,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=int(3200),
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
    ),
)
gym_hybrid_hppo_config = EasyDict(gym_hybrid_hppo_config)
main_config = gym_hybrid_hppo_config

gym_hybrid_hppo_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
gym_hybrid_hppo_create_config = EasyDict(gym_hybrid_hppo_create_config)
create_config = gym_hybrid_hppo_create_config

# if __name__ == "__main__":
#     # or you can enter `ding -m serial -c gym_hybrid_hppo_config.py -s 0`
#     from ding.entry import serial_pipeline_onpolicy
#     serial_pipeline_onpolicy([main_config, create_config], seed=0)

def train(args):
    main_config.exp_name = 'data_hardmove_n10/hppo' + '_seed' + f'{args.seed}'+'_3M'

    serial_pipeline_onpolicy([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed,max_env_step=int(3e6))


if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline_onpolicy

    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)