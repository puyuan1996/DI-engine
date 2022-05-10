from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae

nstep = 3
halfcheetah_dqn_default_config = dict(
    exp_name='halfcheetah_dqn_vqvae_seed0_3M',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        # stop_value=12000,
        stop_value=int(1e6),
    ),
    policy=dict(
        # learned_model_path='/home/puyuan/DI-engine/data_halfcheetah/noobs_noprio_ema_nonoise_rlcipgrad_vhd512_D256_k128seed_0_3M/ckpt/ckpt_best.pth.tar',

        # Whether to use cuda for network.
        cuda=True,
        priority=False,
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        action_space='continuous',  # 'hybrid'

        eps_greedy_nearest=False, # TODO
        is_ema_target=False,  # use EMA

        is_ema=False,  # no use EMA # TODO
        # is_ema=True,  # use EMA
        original_action_shape=6,
        random_collect_size=int(5e4),
        # random_collect_size=int(1),  # debug
        vqvae_embedding_dim=64,  # ved: D
        vqvae_hidden_dim=[256],  # vhd
        vq_loss_weight=0.1,
        model=dict(
            obs_shape=17,
            action_shape=int(64),  # num of num_embeddings: K
            # encoder_hidden_size_list=[128, 128, 64],  # small net
            encoder_hidden_size_list=[256, 256, 128],  # middle net
            # encoder_hidden_size_list=[512, 512, 256],  # large net
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            reconst_loss_stop_value=1e-6, # TODO
            ignore_done=True,  # TODO
            constrain_action=False,  # TODO
            warm_up_update=int(1e4),
            # warm_up_update=int(1),  # debug
            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=20,
            update_per_collect_vae=20,

            rl_batch_size=512,
            vqvae_batch_size=512,

            learning_rate=3e-4,
            learning_rate_vae=3e-4,
            # Frequency of target network update.
            # target_update_theta=0.001,
            target_update_freq=500,
            
            rl_clip_grad=True,
            vqvae_clip_grad=True,
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
halfcheetah_dqn_default_config = EasyDict(halfcheetah_dqn_default_config)
main_config = halfcheetah_dqn_default_config

halfcheetah_dqn_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    # env_manager=dict(type='base'),
    policy=dict(type='dqn_vqvae'),
)
halfcheetah_dqn_create_config = EasyDict(halfcheetah_dqn_create_config)
create_config = halfcheetah_dqn_create_config


import copy

def train(args):
    main_config.exp_name = 'data_halfcheetah/dqn_noema_middlenet_k64_upcr20_vqloss01' + '_seed' + f'{args.seed}'+'_3M'
    serial_pipeline_dqn_vqvae([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(3e6))

if __name__ == "__main__":
    import argparse
    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)
