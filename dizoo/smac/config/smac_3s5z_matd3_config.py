from easydict import EasyDict
from ding.entry import serial_pipeline

agent_num = 8
collector_env_num = 16
evaluator_env_num = 8
special_global_state = True

SMAC_3s5z_matd3_default_config = dict(
    # exp_name='debug_smac_3s5z_matd3',
    env=dict(
        map_name='3s5z',
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
            agent_obs_shape=150,
            global_obs_shape=295,
            action_shape=14,
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
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
            # TODO(pu)
            # alpha=0.2,  
            # auto_alpha=True,
            # entropy_weight=0.01, 
            # log_space=True,
            # TODO(pu)
            alpha=0.,
            auto_alpha=False,
            entropy_weight=0.01, 
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
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ), ),
    ),
)

SMAC_3s5z_matd3_default_config = EasyDict(SMAC_3s5z_matd3_default_config)
main_config = SMAC_3s5z_matd3_default_config

SMAC_3s5z_matd3_default_create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='matd3',
    ),
)
SMAC_3s5z_matd3_default_create_config = EasyDict(SMAC_3s5z_matd3_default_create_config)
create_config = SMAC_3s5z_matd3_default_create_config



def train(args):
    main_config.exp_name='debug_smac_3s5z_matd3'+'_seed'+f'{args.seed}'+'_ew0.01'
    # main_config.exp_name='debug_smac_3s5z_matd3'+'_seed'+f'{args.seed}'+'_clogpi_ew0.01'

    import copy
    # 3125 iterations= 10M/3200 env steps mmm2
    # 6250 iterations= 10M/1600 env steps 5m6m
    # 1562.5 iterations= 5M/3200 env steps mmm
    # 3125 iterations= 5M/1600 env steps 3s5z
    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_iterations=3125)

if __name__ == "__main__":
    import argparse
    for seed in [0,1,2]:     
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        
        train(args)