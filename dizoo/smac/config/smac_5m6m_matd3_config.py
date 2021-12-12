from easydict import EasyDict
from ding.entry import serial_pipeline

agent_num = 5
collector_env_num = 8
evaluator_env_num = 8
special_global_state = True

SMAC_5m6m_matd3_default_config = dict(
    exp_name='debug_smac_5m6m_matd3_d5e4',
    env=dict(
        map_name='5m_vs_6m',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=16,
        stop_value=0.99,
        death_mask=True,  # TODO(pu) False
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
            agent_obs_shape=72,
            global_obs_shape=152,
            action_shape=12,
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            learning_rate_alpha=5e-5,  # TODO(pu)
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            # alpha=0.2,  # TODO(pu)
            # auto_alpha=True,
            alpha=0.,  # TODO(pu)
            auto_alpha=False,
            log_space=True,
        ),
        collect=dict(
            env_num=collector_env_num,
            n_sample=1600,  # TODO（pu)
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(
            evaluator=dict(
                eval_freq=50,
            ),
            env_num=evaluator_env_num,
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=int(5e4),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ), ),
    ),
)

SMAC_5m6m_matd3_default_config = EasyDict(SMAC_5m6m_matd3_default_config)
main_config = SMAC_5m6m_matd3_default_config

SMAC_5m6m_matd3_default_create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='matd3',
    ),
    #replay_buffer=dict(type='naive', ),
)
SMAC_5m6m_matd3_default_create_config = EasyDict(SMAC_5m6m_matd3_default_create_config)
create_config = SMAC_5m6m_matd3_default_create_config


# if __name__ == "__main__":
#     serial_pipeline([main_config, create_config], seed=0)

def train(args):
    main_config.exp_name='debug_smac_5m6m_masac_nologpi'+'_seed'+f'{args.seed}'
    # serial_pipeline([main_config, create_config], seed=args.seed)
    import copy
    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed)

if __name__ == "__main__":
    import argparse
    for seed in [0,1,2]:     
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        
        train(args)