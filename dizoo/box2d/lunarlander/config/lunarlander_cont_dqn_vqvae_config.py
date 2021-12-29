from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae

nstep = 3
lunarlander_dqn_default_config = dict(
    exp_name='debug_lunarlander_cont_dqn_vqvae',
    env=dict(
        env_id='LunarLanderContinuous-v2',
        # (bool) Scale output action into legal range.
        act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        priority=False,
        random_collect_size=10000,
        original_action_shape=2,
        model=dict(
            obs_shape=8,
            action_shape=int(8*8),
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            warm_up_update=int(1e4),
            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=256,
            update_per_collect_vae=10,
            batch_size=128,
            # learning_rate_actor=3e-4,
            # learning_rate_critic=3e-4,
            learning_rate=0.001,
            learning_rate_vae=1e-4,
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=256,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=int(1e5), )
        ),
    ),
)
lunarlander_dqn_default_config = EasyDict(lunarlander_dqn_default_config)
main_config = lunarlander_dqn_default_config

lunarlander_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_vqvae'),
)
lunarlander_dqn_create_config = EasyDict(lunarlander_dqn_create_config)
create_config = lunarlander_dqn_create_config

if __name__ == "__main__":
    serial_pipeline_dqn_vqvae([main_config, create_config], seed=0)
