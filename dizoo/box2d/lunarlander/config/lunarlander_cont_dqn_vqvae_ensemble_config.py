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
        # debug
        # collector_env_num=1,
        # evaluator_env_num=1,
        # n_evaluator_episode=1,
        # stop_value=200,
        stop_value=int(1e6),
        # replay_path='/home/puyuan/DI-engine/data_lunarlander/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_seed1_3M',
        replay_path='/Users/puyuan/code/DI-engine/data_lunarlander_visualize',
        # save_replay_gif=True,
        save_replay_gif=False,
    ),
    policy=dict(
        model_path=None,
        # model_path='/Users/puyuan/code/DI-engine/data_lunarlander/dqn_sbh_ensemble20_noobs_noema_smallnet_k8_upc50_seed1_3M/ckpt/ckpt_best.pth.tar',

        # Whether to use cuda for network.
        cuda=True,

        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        action_space='continuous',
        eps_greedy_nearest=False,  # TODO(pu): delete this key
        is_ema_target=False,

        # TODO(pu): test ema
        is_ema=False,  # no use EMA
        # is_ema=True,  # use EMA
        original_action_shape=2,
        random_collect_size=int(5e4),  # transitions
        warm_up_update=int(1e4),
        # debug
        # random_collect_size=int(10),  
        # warm_up_update=int(2),
        # eval
        # random_collect_size=int(0),
        # warm_up_update=int(0),

        vqvae_embedding_dim=64,  # ved: D
        vqvae_hidden_dim=[256],  # vhd
        target_network_soft_update=False,
        beta=0.25,
        vq_loss_weight=0.1,
        recons_loss_cont_weight=1,
        mask_pretanh=False,
        replay_buffer_size_vqvae=int(1e6),
        auxiliary_conservative_loss=False,
        augment_extreme_action=False,
        # augment_extreme_action=True,
        # TODO
        obs_regularization=True,
        # obs_regularization=False,
        predict_loss_weight=0,  

        # TODO
        # only if obs_regularization=True, this option take effect
        # v_contrastive_regularization=False,
        v_contrastive_regularization=True,
        contrastive_regularization_loss_weight=1,

        # vqvae_pretrain_only=True,
        # NOTE: if only pretrain vqvae , i.e. vqvae_pretrain_only=True, should set this key to False
        # recompute_latent_action=False,

        vqvae_pretrain_only=False,
        # NOTE: if train vqvae dynamically, i.e. vqvae_pretrain_only=False, should set this key to True
        recompute_latent_action=True,

        # optional design
        cont_reconst_l1_loss=False,
        cont_reconst_smooth_l1_loss=False,

        categorical_head_for_cont_action=False,  # categorical distribution

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
        latent_action_shape=int(4),  # num of num_embeddings: K, i.e. shape of latent action
        model=dict(
            ensemble_num=20,  # TODO
            obs_shape=8,
            action_shape=int(4),  # num of num_embeddings, K
            encoder_hidden_size_list=[128, 128, 64],  # small net
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            # NOTE: only halfcheetah, set this key True
            ignore_done=False,

            reconst_loss_stop_value=1e-6,  # TODO(pu)
            constrain_action=False,  # TODO(pu): delete this key
           
            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=50,  # for collector n_sample=256
            update_per_collect_vae=50,

            rl_batch_size=512,
            vqvae_batch_size=512,
            # debug
            # rl_batch_size=5,
            # vqvae_batch_size=5,

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

            rl_linear_lr_scheduler=False,

            # add noise in original continuous action
            noise=False,  # NOTE: if vqvae_pretrain_only=True
            # noise=True,  # NOTE: if vqvae_pretrain_only=False
            noise_sigma=0.,
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
lunarlander_dqn_default_config = EasyDict(lunarlander_dqn_default_config)
main_config = lunarlander_dqn_default_config

lunarlander_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_vqvae'),
)
lunarlander_dqn_create_config = EasyDict(lunarlander_dqn_create_config)
create_config = lunarlander_dqn_create_config


def train(args):
    # main_config.exp_name = 'data_lunarlander_visualize/noobs_k8_upc50'
    main_config.exp_name = 'data_lunarlander/dqn_sbh_ensemble20_obs0_noema_smallnet_k4_upc50_crlw1' + '_seed' + f'{args.seed}' + '_3M'
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
        # obs_shape: 8
        # action_shape: 2

