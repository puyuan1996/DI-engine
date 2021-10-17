from easydict import EasyDict
from ding.entry import serial_pipeline

agent_num = 10
collector_env_num = 8
evaluator_env_num = 8
special_global_state = True

cartpole_sac_default_config = dict(
    exp_name='smac_MMM_ppo',
    env=dict(
        map_name='MMM',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=16,
        stop_value=0.99,
        death_mask=False,
        special_global_state=special_global_state,
        # save_replay_episodes = 1,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        random_collect_size=0,
        model=dict(
            agent_obs_shape=186,
            global_obs_shape=389,
            action_shape=16,
            twin_critic=True,
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=5,
            batch_size=64,
            learning_rate_q=5e-3,
            learning_rate_policy=5e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.01,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=False,
        ),
        collect=dict(
            env_num=8,
            n_sample=256,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(
            evaluator=dict(
                eval_freq=50,
            ),
            env_num=8,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
)

cartpole_sac_default_config = EasyDict(cartpole_sac_default_config)
main_config = cartpole_sac_default_config

cartpole_sac_default_create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='masac',
    ),
    #replay_buffer=dict(type='naive', ),
)
cartpole_sac_default_create_config = EasyDict(cartpole_sac_default_create_config)
create_config = cartpole_sac_default_create_config


if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)