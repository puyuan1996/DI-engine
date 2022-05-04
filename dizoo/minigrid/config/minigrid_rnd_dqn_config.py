from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
minigrid_dqn_rnd_config = dict(
    exp_name='minigrid_empty8_rnd_dqn_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        # MiniGrid env id: 'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0','MiniGrid-DoorKey-16x16-v0'
        env_id='MiniGrid-Empty-8x8-v0',
        max_step=300,
        stop_value=0.96,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        intrinsic_reward_weight=None,
        # means the relative weight of RND intrinsic_reward.
        # If intrinsic_reward_weight=None, we will automatically set it based on
        # the absolute value of the difference between max and min extrinsic reward in the sampled mini-batch
        # please refer to rnd_reward_model for details.
        # Specifically for sparse reward env MiniGrid, in this env,
        # if reach goal, the agent get reward ~1, otherwise 0.
        # We could set the intrinsic_reward_weight approximately equal to the inverse of max_episode_steps.
        intrinsic_reward_rescale=0.001,
        # means the rescale value of RND intrinsic_reward only used when intrinsic_reward_weight is None
        # please refer to rnd_reward_model for details.
        learning_rate=5e-4,
        obs_shape=2739,
        batch_size=320,
        update_per_collect=10,
        clear_buffer_per_iters=10,
        obs_norm=True,
        obs_norm_clamp_max=5,
        obs_norm_clamp_min=-5,
    ),
    policy=dict(
        cuda=True,
        nstep=3,
        discount_factor=0.99,
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=3200, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
minigrid_dqn_rnd_config = EasyDict(minigrid_dqn_rnd_config)
main_config = minigrid_dqn_rnd_config
minigrid_dqn_rnd_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
    reward_model=dict(type='rnd'),
)
minigrid_dqn_rnd_create_config = EasyDict(minigrid_dqn_rnd_create_config)
create_config = minigrid_dqn_rnd_create_config

if __name__ == "__main__":
    # TODO(pu): how to deal with nstep reward in dqn+rnd?
    # or you can enter `ding -m serial_reward_model_offpolicy -c minigrid_rnd_config.py -s 0`
    from ding.entry import serial_pipeline_reward_model_offpolicy
    serial_pipeline_reward_model_offpolicy([main_config, create_config], seed=0)
