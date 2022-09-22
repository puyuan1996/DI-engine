from easydict import EasyDict
import os

# os.environ['DISPLAY'] = ':1'

halfcheetah_sac_config = dict(
    exp_name='halfcheetah_sac_seed0',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        clip_rewards=False,
        # stop_value=12000,
        stop_value=99999,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

halfcheetah_sac_config = EasyDict(halfcheetah_sac_config)
main_config = halfcheetah_sac_config

halfcheetah_sac_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
halfcheetah_sac_create_config = EasyDict(halfcheetah_sac_create_config)
create_config = halfcheetah_sac_create_config

# if __name__ == "__main__":
#     # or you can enter `ding -m serial -c halfcheetah_sac_config.py -s 0`
#     from ding.entry import serial_pipeline
#     serial_pipeline((main_config, create_config), seed=0)

def train(args):
    main_config.exp_name = 'data_halfcheetah/sac_seed' + f'{args.seed}' + '_3M'
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
        # obs_shape: 17
        # action_shape: 6