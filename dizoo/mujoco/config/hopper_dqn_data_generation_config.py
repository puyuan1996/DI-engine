from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae

nstep = 3
hopper_dqn_default_config = dict(
    exp_name='data_hopper/dqn_middlenet_k64_upc20_seed0_3M',
    env=dict(
        env_id='Hopper-v3',
        each_dim_disc_size=4,
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        # collector_env_num=8,
        # evaluator_env_num=8,
        # n_evaluator_episode=8,
        collector_env_num=2,
        evaluator_env_num=2,
        n_evaluator_episode=2,
        # stop_value=12000,
        stop_value=int(1e6),
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        priority=False,
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
                model_path='/home/puyuan/DI-engine/data_hopper/dqn_middlenet_k64_upc20_seed0_3M/ckpt/ckpt_best.pth.tar',
                # load_path='/home/puyuan/DI-engine/data_hopper/dqn_middlenet_k64_upc20_seed0_3M/ckpt/iteration_3102260.pth.tar',
                # hook=dict(
                #     load_ckpt_before_run='/home/puyuan/DI-engine/data_hopper/dqn_middlenet_k64_upc20_seed0_3M/ckpt/ckpt_best.pth.tar',
                #     # load_ckpt_before_run='/home/puyuan/DI-engine/data_hopper/dqn_middlenet_k64_upc20_seed0_3M/ckpt/iteration_3102260.pth.tar',
                #     save_ckpt_after_run=False,
                # )
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
            save_path='/home/puyuan/hopper_dqn_seed0/expert_data.pkl',
            # load
            data_type='naive',
            data_path='/home/puyuan/hopper_dqn_seed0/expert_data.pkl'
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
            replay_buffer=dict(replay_buffer_size=int(2), ) # TODO
        ),
    ),
)
hopper_dqn_default_config = EasyDict(hopper_dqn_default_config)
main_config = hopper_dqn_default_config

hopper_dqn_create_config = dict(
    env=dict(
        type='mujoco-disc',
        import_names=['dizoo.mujoco.envs.mujoco_env_disc'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
hopper_dqn_create_config = EasyDict(hopper_dqn_create_config)
create_config = hopper_dqn_create_config


