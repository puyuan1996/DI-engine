from easydict import EasyDict

collector_env_num=1
# collector_env_num=8
evaluator_env_num=8

num_actuators = 10

gym_hybrid_onppo_default_config = dict(
    exp_name='gym_hybrid_onppo_vqvae_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        # (bool) Scale output action into legal range [-1, 1].
        act_scale=True,
        env_id='Moving-v0',
        # env_id='Sliding-v0',
        # env_id='HardMove-v0',
        num_actuators=num_actuators,  # only for 'HardMove-v0'
        # stop_value=2,
        stop_value=int(1e6),  # stop according to max env steps
        save_replay_gif=False,
        replay_path=None,
    ),
    policy=dict(
        model_path=None,

        # Whether to use cuda for network.
        cuda=True,
        
        # ppo related
        recompute_adv=True,
        action_space='discrete',

        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,

        original_action_space='hybrid',  # 'hybrid'
        eps_greedy_nearest=False,  # TODO(pu): delete this key
        is_ema_target=False,

        is_ema=False,  # no use EMA 
        # is_ema=True,  # use EMA TODO(pu): test ema
        # for 'Moving-v0', 'Sliding-v0'
        original_action_shape=dict(
            action_type_shape=3,
            action_args_shape=2,
        ),
        # for 'HardMove-v0'
        # original_action_shape=dict(
        #     action_type_shape=int(2 ** num_actuators),  # 2**4=16, 2**6=64, 2**8=256, 2**10=1024
        #     action_args_shape=int(num_actuators),  # 4,6,8,10
        # ),
        # random_collect_size=int(1000),  # n_episode
        # warm_up_update=int(1e4),
        # debug
        random_collect_size=int(1),
        warm_up_update=int(1),

        # target_network_soft_update=False,
        # target_network_soft_update=True,

        vqvae_embedding_dim=64,  # ved: D
        vqvae_hidden_dim=[256],  # vhd
        beta=0.25,
        vq_loss_weight=0.1,  # TODO
        recons_loss_cont_weight=1,
        # mask_pretanh=True,
        mask_pretanh=False,
        replay_buffer_size_vqvae=int(1e6),
        auxiliary_conservative_loss=False,
        augment_extreme_action=False,

        # obs_regularization=True,
        obs_regularization=False,
        predict_loss_weight=1,  # TODO

        # vqvae_pretrain_only=True,
        # NOTE: if only pretrain vqvae , i.e. vqvae_pretrain_only=True, should set this key to False
        # recompute_latent_action=False,

        # TODO
        vqvae_pretrain_only=False,
        # NOTE: if train vqvae dynamically, i.e. vqvae_pretrain_only=False, should set this key to True
        recompute_latent_action=True,

        # optional design
        cont_reconst_l1_loss=False,
        cont_reconst_smooth_l1_loss=False,
        categorical_head_for_cont_action=False,  # categorical distribution

        # threshold_categorical_head_for_cont_action=True,  # thereshold categorical distribution
        threshold_categorical_head_for_cont_action=False,  # thereshold categorical distribution
        categorical_head_for_cont_action_threshold=0.9,
        threshold_phase=['eval'],  # ['eval', 'collect']
        n_atom=11,

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
        latent_action_shape=int(16),  # num of num_embeddings: K, i.e. shape of latent action
        # latent_action_shape=int(64),  # num of num_embeddings: K, i.e. shape of latent action
        model=dict(
            action_space='discrete',
            obs_shape=10,  # related to the environment
            # for moving/sliding
            action_shape=int(16),  # num of num_embeddings: K
            encoder_hidden_size_list=[128, 128, 64],  # small net
            actor_head_hidden_size=64,
            critic_head_hidden_size=64,
            # for hardmove
            # action_shape=int(64),  # num of num_embeddings: K
            # encoder_hidden_size_list=[256, 256, 128],  # middle net
            # actor_head_hidden_size= 128,
            # critic_head_hidden_size=128,
        ),
        learn=dict(
            ignore_done=False,

            # ppo related
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,

            reconst_loss_stop_value=1e-6,  # TODO(pu)
            constrain_action=False, # TODO(pu): delete this key

            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=1,
            update_per_collect_vae=1,

            epoch_per_collect_rl=10,
            epoch_per_collect_vqvae=10,
            
            rl_batch_size=256,
            vqvae_batch_size=256,
            learning_rate=3e-4,
            learning_rate_vae=3e-4,
            # Frequency of target network update.
            target_update_freq=500,
            target_update_theta=0.001,

            rl_clip_grad=True,
            vqvae_clip_grad=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            # rl_weight_decay=1e-4,
            # vqvae_weight_decay=1e-4,
            rl_weight_decay=None,
            vqvae_weight_decay=None,

            # add noise in original continuous action
            noise=False,  # NOTE: if vqvae_pretrain_only=True
            # noise=True,  # NOTE: if vqvae_pretrain_only=False
            noise_sigma=0.1,
            noise_range=dict(
            min=-0.5,
            max=0.5,
            ),
            noise_augment_extreme_action=True,
            noise_augment_extreme_action_prob=0.1,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            # TODO
            # n_episode=8,
            n_episode=collector_env_num,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
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
            replay_buffer=dict(replay_buffer_size=int(1e6), ),
        ),
    ),
)
gym_hybrid_onppo_default_config = EasyDict(gym_hybrid_onppo_default_config)
main_config = gym_hybrid_onppo_default_config

gym_hybrid_onppo_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='onppo_vqvae'),
    collector=dict(type='episode', get_train_sample=True)
)
gym_hybrid_onppo_create_config = EasyDict(gym_hybrid_onppo_create_config)
create_config = gym_hybrid_onppo_create_config


def train(args):
    main_config.exp_name = 'data_moving/onppo_noobs_epcr10_noema_smallnet_k16_beta0.25_vlw01' + '_seed' + f'{args.seed}'+'_3M'
    serial_pipeline_onppo_vqvae([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(3e6))

if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline_onppo_vqvae
    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)