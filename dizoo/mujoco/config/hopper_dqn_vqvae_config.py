from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae

nstep = 3
hopper_dqn_default_config = dict(
    exp_name='debug_hopper_dqn_vqvae_ved128_k128_ehsl256256128_upcr20_bs512_ed1e5_rbs1e6_seed0_3M',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        # stop_value=3000,
        stop_value=int(1e6),  # max env steps 
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        # priority=False,
        priority=False,
        # priority=True,
        random_collect_size=int(1e4),
        # random_collect_size=int(1),  # debug

        original_action_shape=3,
        vqvae_embedding_dim=128,  # ved
        vqvae_hidden_dim=[256],  # vhd

        is_ema_target=False,  # use EMA
        is_ema=True,  # use EMA
        # is_ema=False,  # no EMA
        eps_greedy_nearest=True,

        action_space='continuous',  # 'hybrid',
        model=dict(
            obs_shape=11,
            action_shape=int(128),  # num of num_embeddings
            # encoder_hidden_size_list=[128, 128, 64],  # small net
            encoder_hidden_size_list=[256, 256, 128],  # middle net
            # encoder_hidden_size_list=[512, 512, 256],  # large net

            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            constrain_action=False,
            warm_up_update=int(1e4),
            # warm_up_update=int(1), # debug

            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            # update_per_collect_rl=20,
            update_per_collect_rl=256,
            update_per_collect_vae=10,
            rl_batch_size=512,
            vqvae_batch_size=512,
            learning_rate=3e-4,
            learning_rate_vae=1e-4,
            # Frequency of target network update.
            target_update_freq=500,

            # NOTE
            rl_clip_grad=True,
            # rl_clip_grad=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,

            # add noise in original continuous action
            # noise=True,
            noise=False,
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
        eval=dict(evaluator=dict(eval_freq=5000, )),
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
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_vqvae'),
)
hopper_dqn_create_config = EasyDict(hopper_dqn_create_config)
create_config = hopper_dqn_create_config

# if __name__ == "__main__":
#     serial_pipeline_dqn_vqvae([main_config, create_config], seed=0)

import copy

def train(args):
    # main_config.exp_name = 'data_hopper/hopper_ema_upcr20_bs512_' + 'seed_' + f'{args.seed}'
    main_config.exp_name = 'data_hopper/noobs_ema_nonoise_rlclipgrad0.5_noprio_wu1e4_eps-greedy-nearest_' + 'seed' + f'{args.seed}'+'_3M'
    # main_config.exp_name = 'hopper_noema_upcr20_bs32_' + 'seed_' + f'{args.seed}'

    serial_pipeline_dqn_vqvae(
        [copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed
    )  #, max_env_step=int(3e3))


if __name__ == "__main__":
    import argparse
    # for seed in [0, 1, 2, 3, 4]:
    for seed in [0]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)