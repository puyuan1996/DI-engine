from easydict import EasyDict

hopper_sac_data_genearation_default_config = dict(
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=11,
            action_shape=3,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
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
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
            learner=dict(
                load_path='/home/puyuan/hopper_sac_seed0/ckpt/ckpt_best.pth.tar',
            ),
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            # save path
            save_path='/home/puyuan/hopper_sac_seed0/expert_data_1000eps.pkl',
            save_path_transitions='/home/puyuan/hopper_sac_seed0/expert_data_transitions_1000eps.pkl',
            # load path
            data_type='naive',
            data_path='/home/puyuan/hopper_sac_seed0/expert_data_transitions_1000eps_lt3500.pkl'
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000, ), ),
    ),
)

hopper_sac_data_genearation_default_config = EasyDict(hopper_sac_data_genearation_default_config)
main_config = hopper_sac_data_genearation_default_config

hopper_sac_data_genearation_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_sac_data_genearation_default_create_config = EasyDict(hopper_sac_data_genearation_default_create_config)
create_config = hopper_sac_data_genearation_default_create_config
