from easydict import EasyDict
from ding.entry import serial_pipeline

agent_num = 27
collector_env_num = 8
evaluator_env_num = 8
special_global_state = True

SMAC_27m30m_masac_default_config = dict(
    exp_name='debug_smac_27m30m_masac_d5e4',
    env=dict(
        map_name='27m_vs_30m',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=16,
        stop_value=0.99,
        death_mask=True,
        special_global_state=special_global_state,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        # random_collect_size=0,
        random_collect_size=int(1e4),
        model=dict(
            agent_obs_shape=348,
            global_obs_shape=1454,
            action_shape=36,
            twin_critic=True,
            # actor_head_hidden_size=256,
            # critic_head_hidden_size=256,
            actor_head_hidden_size=512,
            critic_head_hidden_size=512,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            learning_rate_alpha=5e-5,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=True,
            log_space=True,
        ),
        collect=dict(
            env_num=collector_env_num,
            n_sample=1600,
            unroll_len=1,
        ),
        eval=dict(
            evaluator=dict(eval_freq=50, ),
            env_num=evaluator_env_num,
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(5e4), ), ),
    ),
)

SMAC_27m30m_masac_default_config = EasyDict(SMAC_27m30m_masac_default_config)
main_config = SMAC_27m30m_masac_default_config

SMAC_27m30m_masac_default_create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sac_discrete', ),
)
SMAC_27m30m_masac_default_create_config = EasyDict(SMAC_27m30m_masac_default_create_config)
create_config = SMAC_27m30m_masac_default_create_config


# if __name__ == "__main__":
#     serial_pipeline([main_config, create_config], seed=0)

def train(args):
    main_config.exp_name='debug_smac_27m30m_masac'+'_seed'+f'{args.seed}'+'_hs512_rcs1e4_rbs5e4_ed1e5'
    import copy
    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed)


if __name__ == "__main__":
    import argparse
    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        
        train(args)
