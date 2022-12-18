from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5
n_evaluator_episode = 5
# debug
# collector_env_num=2
# evaluator_env_num=2
# n_evaluator_episode=2

gym_hybrid_hppo_config = dict(
    exp_name='gym_hybrid_hppo_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        # (bool) Scale output action into legal range, usually [-1, 1].
        act_scale=True,
        env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
        stop_value=1e6,
        save_replay_gif=False,
        replay_path_gif=None,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        action_space='hybrid',
        recompute_adv=True,
        model=dict(
            obs_shape=10,
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
            action_space='hybrid',
            encoder_hidden_size_list=[256, 128, 64, 64],
            sigma_type='fixed',
            fixed_sigma_value=0.3,
            bound_type='tanh',
        ),
        learn=dict(
            continuous_loss_weight=1,
            discrete_loss_weight=0.5,
            epoch_per_collect=10,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.03,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            n_sample=int(3200),
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=1000, n_episode=n_evaluator_episode)),
    ),
)
gym_hybrid_hppo_config = EasyDict(gym_hybrid_hppo_config)
main_config = gym_hybrid_hppo_config

gym_hybrid_hppo_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    # env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
gym_hybrid_hppo_create_config = EasyDict(gym_hybrid_hppo_create_config)
create_config = gym_hybrid_hppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c gym_hybrid_hppo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    main_config.exp_name = "dev/gym_hybrid_hppo_seed0_2head_dlw05"
    serial_pipeline_onpolicy([main_config, create_config], seed=0, max_env_step=int(3e6))
