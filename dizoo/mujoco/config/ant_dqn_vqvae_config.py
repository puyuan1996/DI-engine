from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae

nstep = 3
ant_dqn_default_config = dict(
    exp_name='ant_dqn_vqvae_seed0_3M',
    env=dict(
        env_id='Ant-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        # stop_value=5000,
        stop_value=int(1e6),  # stop according to max env steps
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,

        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        action_space='continuous',  # 'hybrid'
        eps_greedy_nearest=False,  # TODO(pu): delete this key
        is_ema_target=False,

        is_ema=False,  # no use EMA
        # is_ema=True,  # use EMA TODO(pu): test ema
        original_action_shape=8,  # related to the environment
        random_collect_size=int(5e4),  # transitions
        warm_up_update=int(1e4),
        # debug
        # random_collect_size=int(10),
        # warm_up_update=int(10),

        vqvae_embedding_dim=64,  # ved: D
        vqvae_hidden_dim=[256],  # vhd
        # vqvae_hidden_dim=[512],  # vhd
        # vqvae_hidden_dim=[1024],  # vhd
        vq_loss_weight=1,
        replay_buffer_size_vqvae=int(1e6),

        obs_regularization=True,
        # obs_regularization=False,
        predict_loss_weight=1,  # TODO

        # vqvae_pretrain_only=True,
        # NOTE: if only pretrain vqvae , i.e. vqvae_pretrain_only=True, should set this key to False
        # recompute_latent_action=False,

        # TODO
        vqvae_pretrain_only=False,
        # NOTE: if train vqvae dynamically, i.e. vqvae_pretrain_only=False, should set this key to True
        recompute_latent_action=True,

        # optinal design
        cont_reconst_l1_loss=False,
        cont_reconst_smooth_l1_loss=False,
        categorical_head_for_cont_action=False,  # categorical distribution
        n_atom=51,
        gaussian_head_for_cont_action=False,  # gaussian distribution
        embedding_table_onehot=False,

        # rl priority
        priority=False,
        priority_IS_weight=False,
        # TODO: weight RL loss according to the reconstruct loss, because in In the area with large reconstruction
        #  loss, the action reconstruction is inaccurate, that is, the (\hat{x}, r) does not match,
        #  and the corresponding Q value is inaccurate. The update should be reduced to avoid wrong gradient.
        rl_reconst_loss_weight=False,
        rl_reconst_loss_weight_min=0.2,

        # vqvae priority
        vqvae_return_weight=False,  # NOTE: return weight

        priority_vqvae=False,  # NOTE: return priority
        priority_IS_weight_vqvae=False,  # NOTE: return priority
        priority_type_vqvae='return',
        priority_vqvae_min=0.,
        model=dict(
            obs_shape=111,  # related to the environment
            action_shape=int(64),  # num of num_embeddings: K
            # encoder_hidden_size_list=[128, 128, 64],  # small net
            encoder_hidden_size_list=[256, 256, 128],  # middle net
            # encoder_hidden_size_list=[512, 512, 256],  # large net
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            reconst_loss_stop_value=1e-6,  # TODO(pu)
            constrain_action=False,  # TODO(pu): delete this key

            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=20,  # for collector n_sampe=256
            update_per_collect_vae=20,

            rl_batch_size=512,
            vqvae_batch_size=512,
            learning_rate=3e-4,
            learning_rate_vae=3e-4,
            # Frequency of target network update.
            target_update_freq=500,

            rl_clip_grad=True,
            vqvae_clip_grad=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,

            # add noise in original continuous action
            noise=False,  # NOTE: if vavae_pretrain_only=True
            # noise=True,  # NOTE: if vavae_pretrain_only=False
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
ant_dqn_default_config = EasyDict(ant_dqn_default_config)
main_config = ant_dqn_default_config

ant_dqn_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn_vqvae'),
)
ant_dqn_create_config = EasyDict(ant_dqn_create_config)
create_config = ant_dqn_create_config


import copy
def train(args):
    main_config.exp_name = 'data_ant/ant_obs_noema_middlenet_k64_' + 'seed' + f'{args.seed}'
    serial_pipeline_dqn_vqvae([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed,
                              max_env_step=int(3e6))


if __name__ == "__main__":
    import argparse
    # for seed in [0, 1, 2, 3, 4]:
    for seed in [0]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)
