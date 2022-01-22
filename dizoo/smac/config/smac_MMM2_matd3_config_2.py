from easydict import EasyDict
from ding.entry import serial_pipeline

agent_num = 10
collector_env_num = 8
evaluator_env_num = 8
special_global_state = True

SMAC_MMM2_masac_default_config = dict(
    exp_name='debug_smac_MMM2_masac',
    env=dict(
        map_name='MMM2',
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
        random_collect_size=0,
        model=dict(
            agent_obs_shape=204,
            global_obs_shape=431,
            action_shape=18,
            twin_critic=True,
            # actor_head_hidden_size=256,
            # critic_head_hidden_size=256,
            actor_head_hidden_size=512,
            critic_head_hidden_size=1024,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            learning_rate_alpha=5e-5,
            ignore_done=False,
            target_theta=0.005,
            # discount_factor=0.99,
            discount_factor=0.97,
            # TODO(pu)
            # alpha=0.2,  
            # auto_alpha=True,
            # entropy_weight=0., 
            # log_space=True,
            # TODO(pu)
            alpha=0.,
            auto_alpha=False,
            entropy_weight=0., 
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
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=1000000, ),
        ),
    ),
)

SMAC_MMM2_masac_default_config = EasyDict(SMAC_MMM2_masac_default_config)
main_config = SMAC_MMM2_masac_default_config

SMAC_MMM2_masac_default_create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sac_discrete', ),
)
SMAC_MMM2_masac_default_create_config = EasyDict(SMAC_MMM2_masac_default_create_config)
create_config = SMAC_MMM2_masac_default_create_config



# if __name__ == "__main__":
#     serial_pipeline([main_config, create_config], seed=0)

def train(args):
    main_config.exp_name='debug_smac_MMM2_matd3'+'_seed'+f'{args.seed}'+'_ew0'
    # main_config.exp_name='debug_smac_MMM2_matd3'+'_seed'+f'{args.seed}'+'_clogpi_ew0'
    # main_config.exp_name='debug_smac_MMM2_matd3'+'_seed'+f'{args.seed}'+'_alogpi'

    import copy
    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed)


if __name__ == "__main__":
    import argparse

    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)