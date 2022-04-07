from easydict import EasyDict

halfcheetah_sac_data_genearation_default_config = dict(
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=10,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=12000,  # TODO
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            # halfcheetah # TODO
            obs_shape=17,
            action_shape=6,
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
            # halfcheetah # TODO
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
            learner=dict(
                load_path='/home/puyuan/halfcheetah_sac_seed0/ckpt/ckpt_best.pth.tar',
                hook=dict(
                    load_ckpt_before_run='/home/puyuan/halfcheetah_sac_seed0/ckpt/ckpt_best.pth.tar',
                    save_ckpt_after_run=False,
                )
            ),
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            # save
            # save_path='/home/puyuan/halfcheetah_sac_seed0/expert_iteration_200000.pkl',
            save_path='/home/puyuan/halfcheetah_sac_seed0/expert_data.pkl',
            # load
            data_type='naive',
            data_path='/home/puyuan/halfcheetah_sac_seed0/expert_data.pkl'
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1, ), ),  # TODO
    ),
)

halfcheetah_sac_data_genearation_default_config = EasyDict(halfcheetah_sac_data_genearation_default_config)
main_config = halfcheetah_sac_data_genearation_default_config

halfcheetah_sac_data_genearation_default_create_config = dict(
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
halfcheetah_sac_data_genearation_default_create_config = EasyDict(halfcheetah_sac_data_genearation_default_create_config)
create_config = halfcheetah_sac_data_genearation_default_create_config
