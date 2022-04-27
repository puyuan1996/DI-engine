from easydict import EasyDict
import os
module_path = os.path.dirname(__file__)

nstep = 3
lunarlander_dqn_default_config = dict(
    exp_name='lunarlander_cont_dqn_vqvae_seed0',
    env=dict(
        env_id='LunarLanderContinuous-v2',
        # (bool) Scale output action into legal range.
        act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=200,
        # stop_value=int(1e6),
    ),
    policy=dict(
        # learned_model_path=module_path + '/learned_model_path/iteration_0.pth.tar',  # TODO(pu)

        # Whether to use cuda for network.
        cuda=True,
        priority=False,
        random_collect_size=int(1e4),
        # random_collect_size=int(1),  # debug
        original_action_shape=2,
        vqvae_embedding_dim=64,  # ved
        vqvae_hidden_dim=[256],  # vhd
        vq_loss_weight=0.1,  # TODO
        is_ema=True,  # use EMA
        # is_ema=False,  # use EMA

        is_ema_target=False, 
        # eps_greedy_nearest=False,
        eps_greedy_nearest=True,

        action_space='continuous',  # 'hybrid',
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        model=dict(
            obs_shape=8,
            action_shape=int(64),  # num of num_embeddings, K
            encoder_hidden_size_list=[128, 128, 64],  # small net
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            constrain_action=False,
            warm_up_update=int(1e4),
            # warm_up_update=int(1),  # debug
            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=256,
            update_per_collect_vae=10,
            rl_batch_size=512,
            vqvae_batch_size=512,
            # rl_batch_size=32,
            # vqvae_batch_size=32,

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
                # decay=int(1e5),
                decay=int(1e4),

            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), )
        ),
    ),
)
lunarlander_dqn_default_config = EasyDict(lunarlander_dqn_default_config)
main_config = lunarlander_dqn_default_config

lunarlander_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_vqvae'),
)
lunarlander_dqn_create_config = EasyDict(lunarlander_dqn_create_config)
create_config = lunarlander_dqn_create_config


def train(args):
    main_config.exp_name = 'data_lunarlander/ema_egn_rlclipgrad0.5_vq1_ed1e4' + '_seed' + f'{args.seed}'+'_3M'
    # main_config.exp_name = 'data_lunarlander/noema_rlclipgrad0.5_vq0.1_ed1e4' + '_seed' + f'{args.seed}'+'_3M'
    serial_pipeline_dqn_vqvae([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(3e6))

if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline_dqn_vqvae
    for seed in [0]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        train(args)
