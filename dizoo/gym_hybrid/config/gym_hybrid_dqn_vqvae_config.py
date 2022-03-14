from easydict import EasyDict
from ding.entry import serial_pipeline_dqn_vqvae
import os
module_path = os.path.dirname(__file__)

nstep = 3
gym_hybrid_dqn_default_config = dict(
    exp_name='debug_gym_hybrid_cont_dqn_vqvae_ved64_k64_ehsl12812864_upcr256_bs512_ed1e5_rbs1e6_seed0_3M',

    env=dict(
        # collector_env_num=8,
        # evaluator_env_num=5,
        collector_env_num=1,
        evaluator_env_num=1,
        # (bool) Scale output action into legal range [-1, 1].
        act_scale=True,
        # env_id='Moving-v0',  # ['Moving-v0', 'Sliding-v0']
        env_id='Sliding-v0',  # ['Moving-v0', 'Sliding-v0']
        n_evaluator_episode=5,
        stop_value=2,
    ),
    policy=dict(
        # learned_model_path=module_path + '/learned_model_path/dqn_vqvae_k64_ckpt_best.pth.tar',  # TODO(pu)
        # learned_model_path='/home/puyuan/DI-engine/debug_gym_hybrid_cont_dqn_vqvae_ved64_k64_seed1/ckpt/iteration_38728.pth.tar',

        # Whether to use cuda for network.
        cuda=True,
        # cuda=False,
        priority=True,
        # priority=False,
        random_collect_size=int(1e4),
        # random_collect_size=int(0),  # debug

        # original_action_shape=2,
        original_action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
        vqvae_embedding_dim=64,  # ved
        is_ema=True,  # use EMA
        # is_ema=False,  # no EMA
        action_space='hybrid',
        model=dict(
            # action_space='hybrid',
            obs_shape=10,
            action_shape=int(64),  # num oof num_embeddings, K
            encoder_hidden_size_list=[128, 128, 64],  # small net
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
            # warm_up_update=int(0), # debug

            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            # update_per_collect_rl=20,
            update_per_collect_rl=256,
            # update_per_collect_rl=64, # nature dqn

            update_per_collect_vae=10,
            rl_batch_size=512,
            vqvae_batch_size=512,

            # batch_size=32, # nature dqn

            learning_rate=3e-4,
            learning_rate_vae=1e-4,
            # Frequency of target network update.
            target_update_freq=500,
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
                start=1,
                end=0.05,
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), )
        ),
    ),
)
gym_hybrid_dqn_default_config = EasyDict(gym_hybrid_dqn_default_config)
main_config = gym_hybrid_dqn_default_config

gym_hybrid_dqn_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_vqvae'),
)
gym_hybrid_dqn_create_config = EasyDict(gym_hybrid_dqn_create_config)
create_config = gym_hybrid_dqn_create_config

# if __name__ == "__main__":
#     serial_pipeline_dqn_vqvae([main_config, create_config], seed=0)

import copy

def train(args):
    # main_config.exp_name = 'data_gym_hybrid/sliding_ema_noobs_upcr256_rlbs512_vqvaebs512_prio_' + 'seed_' + f'{args.seed}'
    main_config.exp_name = 'debug'

    serial_pipeline_dqn_vqvae(
        [copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed
    )


if __name__ == "__main__":
    import argparse
    for seed in [0, 1, 2, 3, 4]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)
