from easydict import EasyDict

hopper_td3_data_genearation_default_config = dict(
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=10,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            twin_critic=True,
            obs_shape=11,
            action_shape=3,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.2,
            noise_range={'min': -0.5, 'max': 0.5},
            learner=dict(
                load_path='/home/puyuan/hopper_td3_seed0/ckpt/ckpt_best.pth.tar',
                hook=dict(
                    load_ckpt_before_run='/home/puyuan/hopper_td3_seed0/ckpt/ckpt_best.pth.tar',
                    save_ckpt_after_run=False,
                )
            ),
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            ### save 
            # save_path='/home/puyuan/hopper_td3_seed0/expert_iteration_200000.pkl',
            save_path='/home/puyuan/hopper_td3_seed0/expert_data_100eps.pkl',
            # save_path_transitions='/home/puyuan/hopper_td3_seed0/expert_data_transitions_100eps.pkl',
            ### load
            data_type='naive',
            # data_path='/home/puyuan/hopper_td3_seed0/expert_data_transitions_1000eps_lt3500.pkl'
            data_path='/home/puyuan/hopper_td3_seed0/expert_data_100eps.pkl'
            # data_path='/home/puyuan/hopper_td3_seed0/expert_data_1000eps.pkl',
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=100, ), ),
    ),
)

hopper_td3_data_genearation_default_config = EasyDict(hopper_td3_data_genearation_default_config)
main_config = hopper_td3_data_genearation_default_config

hopper_td3_data_genearation_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='td3',
        import_names=['ding.policy.td3'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_td3_data_genearation_default_create_config = EasyDict(hopper_td3_data_genearation_default_create_config)
create_config = hopper_td3_data_genearation_default_create_config
