from easydict import EasyDict

nstep = 3
hopper_dqn_default_config = dict(
    exp_name='hopper_dqn_vqvae_seed0',
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
        # stop_value=3000,
        stop_value=int(1e6),  # max env steps 
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
        action_space='continuous',  # 'hybrid',
        original_action_shape=3,

        vqvae_hidden_dim=[256],  # vhd
        vq_loss_weight=1,  # TODO
        is_ema_target=False,  # use EMA

        random_collect_size=int(1e4),
        # random_collect_size=int(1e3),  # debug
        vqvae_embedding_dim=64,  # ved
        # vqvae_embedding_dim=128,  # ved

        # is_ema=True,  # use EMA
        is_ema=False,  # use EMA # TODO
        eps_greedy_nearest=False, # TODO
        # eps_greedy_nearest=True, 

        model=dict(
            obs_shape=11,
            action_shape=int(16),  # num of num_embeddings
            encoder_hidden_size_list=[128, 128, 64],  # small net
            # encoder_hidden_size_list=[256, 256, 128],  # middle net
            # encoder_hidden_size_list=[512, 512, 256],  # large net
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            constrain_action=False,  # TODO
            warm_up_update=int(1e4),
            # warm_up_update=int(200), # debug

            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=256,  # TODO
            # update_per_collect_rl=20,
            update_per_collect_vae=10,
            rl_batch_size=512,
            vqvae_batch_size=512,
            learning_rate=3e-4,
            learning_rate_vae=1e-4,
            # Frequency of target network update.
            target_update_freq=500,
            target_update_theta=0.001,

            # NOTE
            rl_clip_grad=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,

            # add noise in original continuous action
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
            # n_sample=1024,
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
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn_vqvae'),
)
hopper_dqn_create_config = EasyDict(hopper_dqn_create_config)
create_config = hopper_dqn_create_config


def train(args):
    # TODO
    main_config.exp_name = 'data_hopper/noema_rlclipgrad0.5_hardtarget_vq1_ed1e5_smallnet_K16_upcr256' + '_seed' + f'{args.seed}'+'_3M'
    serial_pipeline_dqn_vqvae([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(3e6))

if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline_dqn_vqvae

    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)