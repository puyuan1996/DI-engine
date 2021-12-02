from easydict import EasyDict
from ding.entry import serial_pipeline

gym_soccer_ddpg_config = dict(
    exp_name='gym_soccer_ddpg_seed0',
    env=dict(
        collector_env_num=1,#8,
        evaluator_env_num=1,#5,
        # (bool) Scale output action into legal range [-1, 1].
        act_scale=True,
        env_id='Soccer-v0',
        n_evaluator_episode=5,
        stop_value=1,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        random_collect_size=0,  # soccer action space not support random collect now
        action_space='hybrid',
        model=dict(
            obs_shape=59,
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
            twin_critic=False,
            actor_head_type='hybrid',
        ),
        learn=dict(
            action_space='hybrid',
            update_per_collect=10,  # [5, 10]
            batch_size=32,
            discount_factor=0.99,
            learning_rate_actor=0.0003,  # [0.001, 0.0003]
            learning_rate_critic=0.001,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=32,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.1,
                decay=100000,  # [50000, 100000]
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
gym_soccer_ddpg_config = EasyDict(gym_soccer_ddpg_config)
main_config = gym_soccer_ddpg_config

gym_soccer_ddpg_create_config = dict(
    env=dict(
        type='gym_soccer',
        import_names=['dizoo.gym_soccer.envs.gym_soccer_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ddpg'),
)
gym_soccer_ddpg_create_config = EasyDict(gym_soccer_ddpg_create_config)
create_config = gym_soccer_ddpg_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
