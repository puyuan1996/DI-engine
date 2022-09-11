from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae

nstep = 3
hopper_dqn_default_config = dict(
    exp_name='dqnvqvae_noema_middlenet_k64_pretrainonly_expert_lt3500_seed1_3M',
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
        # collector_env_num=2,
        # evaluator_env_num=2,
        # n_evaluator_episode=2,
        # stop_value=12000,
        stop_value=int(1e6),
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        random_collect_size=int(5e4),
        original_action_shape=3,
        vqvae_embedding_dim=64,  # ved: D
        vqvae_hidden_dim=[256],  # vhd
        is_ema_target=False,  # use EMA
        # is_ema=True,  # use EMA
        is_ema=False,  # no EMA
        eps_greedy_nearest=False,

        action_space='continuous',  # 'hybrid',
        vq_loss_weight=0.1,
        model=dict(
            obs_shape=11,
            action_shape=int(64),  # num of num_embeddings: k
            encoder_hidden_size_list=[256, 256, 128],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        replay_buffer_size_vqvae=int(1e6), # TODO
        priority=False,
        priority_IS_weight=False,
        # TODO: weight RL loss according to the reconstruct loss, because in 
        # In the area with large reconstruction loss, the action reconstruction is inaccurate, that is, the (\hat{x}, r) does not match, 
        # and the corresponding Q value is inaccurate. The update should be reduced to avoid wrong gradient.
        rl_reconst_loss_weight=False,
        # rl_reconst_loss_weight=True,
        rl_reconst_loss_weight_min=0.2,
        # vqvae_return_weight=False,  # NOTE
        # priority_vqvae=True,  # NOTE
        vqvae_return_weight=False,  # NOTE
        priority_vqvae=False,  # NOTE
        priority_IS_weight_vqvae=False,
        priority_type_vqvae='return',
        # priority_type_vqvae='reward',
        priority_vqvae_min=0.,
        cont_reconst_l1_loss=False,
        cont_reconst_smooth_l1_loss=False,
        vavae_pretrain_only=False, # NOTE
        recompute_latent_action=False,
        categorical_head_for_cont_action=False,  # categorical distribution
        n_atom=51,
        gaussian_head_for_cont_action=False, # gaussian  distribution
        embedding_table_onehot=False,
        vqvae_expert_only=False,
        # learn_mode config
        learn=dict(
            reconst_loss_stop_value=1e-06,
            constrain_action=False,
            ignore_done=False,
            warm_up_update=int(1e4),
            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=20,
            update_per_collect_vae=20,
            rl_batch_size=512,
            vqvae_batch_size=512,
            learning_rate=3e-4,
            learning_rate_vae=3e-4,
            # Frequency of target network update.
            target_update_freq=500,
            
            # NOTE
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
            learner=dict(
                model_path='/home/puyuan/DI-engine/data_hopper/dqnvqvae_noema_middlenet_k64_pretrainonly_expert_lt3500_seed1_3M/ckpt/ckpt_best.pth.tar',
                # model_path='/home/puyuan/DI-engine/data_hopper/dqnvqvae_noema_middlenet_k64_pretrainonly_expert_lt3500_seed1_3M/ckpt/iteration_236440.pth.tar',
                # model_path='/home/puyuan/DI-engine/data_hopper/dqnvqvae_noema_middlenet_k64_pretrainonly_expert_lt3500_seed1_3M/ckpt/iteration_100000.pth.tar',

            ),
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=256,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,

            # save
            # save_path='/home/puyuan/DI-engine/data_show/dqnvqvae_noema_middlenet_k64_pretrainonly_expert_lt3500_seed1_3M/data_iteration_236440.pkl',
            # save_path='/home/puyuan/DI-engine/data_show/dqnvqvae_noema_middlenet_k64_pretrainonly_expert_lt3500_seed1_3M/data_iteration_100000.pkl',
            save_path='/home/puyuan/DI-engine/data_show/dqnvqvae_noema_middlenet_k64_pretrainonly_expert_lt3500_seed1_3M/data_best.pkl',
            # load
            data_type='naive',
            data_path='/home/puyuan/DI-engine/data_show/dqnvqvae_noema_middlenet_k64_pretrainonly_expert_lt3500_seed1_3M/data_iteration_236440.pkl'
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
            replay_buffer=dict(replay_buffer_size=int(8), ) # TODO
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

