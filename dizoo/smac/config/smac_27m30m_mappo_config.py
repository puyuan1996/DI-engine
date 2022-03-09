import sys
from copy import deepcopy
from ding.entry import serial_pipeline_onpolicy
from easydict import EasyDict

agent_num = 27
# collector_env_num = 8
# evaluator_env_num = 8
collector_env_num = 1
evaluator_env_num = 1
special_global_state = True,

main_config = dict(
    exp_name='smac_27m30m_ppo',
    env=dict(
        map_name='27m_vs_30m',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=16,
        # n_evaluator_episode=2,
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
        multi_agent=True,
        action_space='discrete',
        model=dict(
            # (int) agent_num: The number of the agent.
            # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
            agent_num=agent_num,
            # (int) obs_shape: The shapeension of observation of each agent.
            # For 3s5z, obs_shape=150; for 2c_vs_64zg, agent_num=404.
            # (int) global_obs_shape: The shapeension of global observation.
            # For 3s5z, obs_shape=216; for 2c_vs_64zg, agent_num=342.
            agent_obs_shape=348,
            #global_obs_shape=216,
            global_obs_shape=1454,
            # (int) action_shape: The number of action which each agent can take.
            # action_shape= the number of common action (6) + the number of enemies.
            # For 3s5z, obs_shape=14 (6+8); for 2c_vs_64zg, agent_num=70 (6+64).
            action_shape=36,
            # (List[int]) The size of hidden layer
            # hidden_size_list=[64],
            # encoder_hidden_size_list= [512, 512, 256], # largenet      
            # actor_head_hidden_size=256,
            # critic_head_hidden_size=256,
            # encoder_hidden_size_list= [1024, 1024, 512],   # largenet2           
            # actor_head_hidden_size=512,
            # critic_head_hidden_size=512,
            
            actor_head_hidden_size=512, # 2layer
            critic_head_hidden_size=1024,

        ),
        # used in state_num of hidden_state
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            epoch_per_collect=10,
            batch_size=3200,
            learning_rate=5e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.05,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
            ignore_done=False,
        ),
        collect=dict(env_num=collector_env_num, n_sample=3200),
        eval=dict(env_num=evaluator_env_num),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)


# if __name__ == "__main__":
#     serial_pipeline([main_config, create_config], seed=0)

def train(args):
    # main_config.exp_name='debug_smac_27m30m_mappo_'+'largenet_'+'seed_'+f'{args.seed}'
    main_config.exp_name='debug_smac_27m30m_mappo_'+'2hlayers_'+'seed_'+f'{args.seed}'
    # serial_pipeline([main_config, create_config], seed=args.seed)
    import copy
    serial_pipeline_onpolicy([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed)


if __name__ == "__main__":
    import argparse
    for seed in [0,1,2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()
        
        train(args)
