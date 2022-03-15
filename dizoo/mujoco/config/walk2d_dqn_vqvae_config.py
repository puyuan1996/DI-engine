from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae

nstep = 3
walker2d_dqn_default_config = dict(
    exp_name='debug_walker2d_dqn_vqvae_ved128_k128_ehsl256256128_upcr20_bs512_ed1e5_rbs1e6_seed0_3M',
    env=dict(
        env_id='Walker2d-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        # stop_value=5000,
        stop_value=int(1e6),
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        # priority=False,
        priority=True,

        random_collect_size=int(1e4),
        original_action_shape=6,
        vqvae_embedding_dim=128,  # ved
        is_ema=True,  # use EMA
        # is_ema=False,  # no EMA
        action_space='continuous',  # 'hybrid',
        model=dict(
            obs_shape=17,
            action_shape=int(128),  # num oof num_embeddings
            encoder_hidden_size_list=[256, 256, 128],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            ignore_done=True,
            warm_up_update=int(1e4),
            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            # update_per_collect_rl=20,
            update_per_collect_rl=256,

            update_per_collect_vae=10,
            # batch_size=512,  # large batch size
            rl_batch_size=512,
            vqvae_batch_size=512,

            learning_rate=3e-4,
            learning_rate_vae=1e-4,
            # Frequency of target network update.
            target_update_freq=500,
            # add noise in original continuous action
            noise=True,
            # noise=False,
            noise_sigma=0.1,
            noise_range=dict(
            min=-0.5,
            max=0.5,
            ),
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
                type='exp',
                start=1,
                end=0.05,
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), )
        ),
    ),
)
walker2d_dqn_default_config = EasyDict(walker2d_dqn_default_config)
main_config = walker2d_dqn_default_config

walker2d_dqn_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_vqvae'),
)
walker2d_dqn_create_config = EasyDict(walker2d_dqn_create_config)
create_config = walker2d_dqn_create_config

# if __name__ == "__main__":
#     serial_pipeline_dqn_vqvae([main_config, create_config], seed=0)

import copy

def train(args):
    main_config.exp_name = 'data_walker2d/walker2d_ema_noobs_upcr256_rlbs512_vqvaebs512_prio_noise_' + 'seed_' + f'{args.seed}'
    # main_config.exp_name = 'debug'  # debug

    serial_pipeline_dqn_vqvae(
        [copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed
    )


if __name__ == "__main__":
    import argparse
    # for seed in [0, 1, 2, 3, 4]:
    for seed in [0]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)
