from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae

nstep = 3
hopper_dqn_default_config = dict(
    exp_name='debug_hopper_dqn_vqvae_ved64_k256_ed1e5_rbs1e6',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        # act_scale=True,
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=3000,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        priority=False,
        random_collect_size=int(1e4),
        original_action_shape=3,
        vqvae_embedding_dim=64,
        # vqvae_embedding_dim=128,
        model=dict(
            obs_shape=11,
            action_shape=int(256),  # num oof num_embeddings
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
                # decay=50000,
                decay=int(1e5),
            ),
            # replay_buffer=dict(replay_buffer_size=int(1e5), )
            replay_buffer=dict(replay_buffer_size=int(1e6), )

        ),
    ),
)
hopper_dqn_default_config = EasyDict(hopper_dqn_default_config)
main_config = hopper_dqn_default_config

hopper_dqn_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_vqvae'),
)
hopper_dqn_create_config = EasyDict(hopper_dqn_create_config)
create_config = hopper_dqn_create_config

if __name__ == "__main__":
    serial_pipeline_dqn_vqvae([main_config, create_config], seed=0)
