from easydict import EasyDict

import os
module_path = os.path.dirname(__file__)

collector_env_num = 4
evaluator_env_num = 4
expert_replay_buffer_size = int(5e3)  # TODO(pu)
"""agent config"""
pong_r2d3_config = dict(
    exp_name='pong_r2d3_offppoexpert_k0_pho1-4_rbs2e4_ds5e3_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        discount_factor=0.997,
        burnin_step=2,
        nstep=5,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <burnin_step> + <unroll_len>
        unroll_len=40,
        learn=dict(
            # according to the r2d3 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect 32 sequence
            # samples, the length of each samlpe sequence is <burnin_step> + <unroll_len>,
            # which is 100 in our seeting, 32*100/400=8, so we set update_per_collect=8
            # in most environments
            value_rescale=True,
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
            # DQFD related parameters
            lambda1=1.0,  # n-step return
            lambda2=1,  # 1.0,  # supervised loss
            lambda3=1e-5,  # 1e-5,  # L2  it's very important to set Adam optimizer optim_type='adamw'.
            lambda_one_step_td=1,  # 1-step return
            margin_function=0.8,  # margin function in JE, here we implement this as a constant
            per_train_iter_k=0,  # TODO(pu)
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            each_iter_n_sample=32,
            env_num=collector_env_num,
            # The hyperparameter pho, the demo ratio, control the propotion of data coming\
            # from expert demonstrations versus from the agent's own experience.
            pho=1 / 4,  # TODO(pu)
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=1e5,
            ),
            replay_buffer=dict(
                replay_buffer_size=
                20000,  # TODO(pu) sequence_length 42 10000 obs need 11GB memory, if rbs=20000, at least 140gb
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
pong_r2d3_config = EasyDict(pong_r2d3_config)
main_config = pong_r2d3_config
pong_r2d3_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d3'),
)
pong_r2d3_create_config = EasyDict(pong_r2d3_create_config)
create_config = pong_r2d3_create_config
"""export config"""
expert_pong_r2d3_config = dict(
    exp_name='expert_pong_r2d3_ppoexpert_k0_pho1-4_rbs2e4_ds5e3_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[64, 64, 128],  # ppo expert policy
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        discount_factor=0.997,
        burnin_step=20,
        nstep=5,
        learn=dict(
            expert_replay_buffer_size=expert_replay_buffer_size,  # TODO(pu)
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            each_iter_n_sample=32,
            # Users should add their own path here (path should lead to a well-trained model)
            # demonstration_info_path='dizoo/atari/config/serial/pong/demo_path/ppo-off_iteration_16127.pth.tar',
            demonstration_info_path=module_path + 'demonstration_info_path_placeholder',
            # Users should add their own  path here. 
            # Absolute path is recommended.
            # demonstration_info_path=module_path + 'demonstration_info_path_placeholder',
            # Cut trajectories into pieces with length "unroll_len". should set as self._unroll_len_add_burnin_step of r2d2
            unroll_len=42,  # TODO(pu) should equals self._unroll_len_add_burnin_step in r2d2 policy
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=expert_replay_buffer_size,  # TODO(pu)
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            ),
        ),
    ),
)
expert_pong_r2d3_config = EasyDict(expert_pong_r2d3_config)
expert_main_config = expert_pong_r2d3_config
expert_pong_r2d3_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo_offpolicy_collect_traj'),
)
expert_pong_r2d3_create_config = EasyDict(expert_pong_r2d3_create_config)
expert_create_config = expert_pong_r2d3_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c pong_r2d3_ofppoexpert_config.py -s 0`
    from ding.entry import serial_pipeline_r2d3
    serial_pipeline_r2d3([main_config, create_config], [expert_main_config, expert_create_config], seed=0)
