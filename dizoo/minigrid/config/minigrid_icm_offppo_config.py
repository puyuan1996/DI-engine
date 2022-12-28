from easydict import EasyDict

minigrid_icm_offppo_config = dict(
    exp_name='minigrid_fourroom_icm_offppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        # minigrid env id: 'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0','MiniGrid-DoorKey-16x16-v0','MiniGrid-AKTDT-7x7-1-v0'
        env_id='MiniGrid-AKTDT-7x7-1-v0',
        max_step=100,
        stop_value=0.96,
        # stop_value=12,  # run fixed env_steps for MiniGrid-AKTDT-7x7-1-v0
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        # intrinsic_reward_weight means the relative weight of RND intrinsic_reward.
        # Specifically for sparse reward env MiniGrid, in this env,
        # if reach goal, the agent get reward ~1, otherwise 0,
        # We could set the intrinsic_reward_weight approximately equal to the inverse of max_episode_steps.
        # Please refer to rnd_reward_model for details.
        intrinsic_reward_weight=0.001,
        learning_rate=3e-4,
        obs_shape=2835,
        batch_size=320,
        update_per_collect=50,
        clear_buffer_per_iters=int(1e3),
        obs_norm=True,
        obs_norm_clamp_max=5,
        obs_norm_clamp_min=-5,
        extrinsic_reward_norm=True,
        extrinsic_reward_norm_max=1,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=2835,
            action_shape=7,
            action_space='discrete',
            encoder_hidden_size_list=[256, 128, 64, 64],
            critic_head_hidden_size=64,
            actor_head_hidden_size=64,
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=200, )),
    ),
)
minigrid_icm_offppo_config = EasyDict(minigrid_icm_offppo_config)
main_config = minigrid_icm_offppo_config
minigrid_icm_offppo_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='icm'),
)
minigrid_icm_offppo_create_config = EasyDict(minigrid_icm_offppo_create_config)
create_config = minigrid_icm_offppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c minigrid_icm_config.py -s 0`
    from ding.entry import serial_pipeline_reward_model_offpolicy
    serial_pipeline_reward_model_offpolicy([main_config, create_config], seed=0, max_env_step=int(10e6))