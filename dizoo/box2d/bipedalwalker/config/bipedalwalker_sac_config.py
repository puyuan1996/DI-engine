from easydict import EasyDict
from ding.entry import serial_pipeline

bipedalwalker_sac_config = dict(
    exp_name='bipedalwalker_sac_seed0',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,
        rew_clip=True,
        replay_path=None,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        random_collect_size=1000,
        model=dict(
            obs_shape=24,
            action_shape=4,
            twin_critic=True,
            action_space='reparameterization',
            # actor_head_hidden_size=128,
            # critic_head_hidden_size=128,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            # update_per_collect=1,
            # batch_size=128,
            update_per_collect=20,
            batch_size=512,
            learning_rate_q=3e-4,
            learning_rate_policy=3e-4,
            learning_rate_alpha=3e-4,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
            value_network=False,
        ),
        collect=dict(
            # n_sample=128,
            n_sample=256,
            unroll_len=1,
        ),
        # other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=int(1e6), ), ),
    ),
)
bipedalwalker_sac_config = EasyDict(bipedalwalker_sac_config)
main_config = bipedalwalker_sac_config
bipedalwalker_sac_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
