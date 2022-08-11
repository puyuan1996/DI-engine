from easydict import EasyDict

nstep = 3
hopper_onppo_default_config = dict(
    exp_name='hopper_onppo_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        # stop_value=4000,
        stop_value=int(1e6),  # stop according to max env steps 
        each_dim_disc_size=4,  # n: discrete size of each dim in origin continuous action
    ),
    policy=dict(
        # model_path='/home/puyuan/DI-engine/data_hopper/onppo_middlenet_k64_upc20_seed0_3M/ckpt/ckpt_best.pth.tar',
        # Whether to use cuda for network.
        cuda=True,
        # ppo related
        recompute_adv=True,
        action_space='discrete',

        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        model=dict(
            obs_shape=11,
            # NOTE：original_action_shape m=3, 
            action_shape=int(64),  # num of num_embeddings: K = n**m e.g. 4**3=64

            # encoder_hidden_size_list=[128, 128, 64],  # small net
            encoder_hidden_size_list=[256, 256, 128],  # middle net
            # encoder_hidden_size_list=[512, 512, 256],  # large net
            
            # ppo related
            action_space='discrete',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            # ppo related
            epoch_per_collect=10,
            update_per_collect=1,

            batch_size=512,
            learning_rate=3e-4,
            
            # ppo related
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            # for onppo, when we recompute adv, we need the key done in data to split traj, so we must
            # use ignore_done=False here,
            # but when we add key traj_flag in data as the backup for key done, we could choose to use ignore_done=True
            # for halfcheetah, the length=1000
            ignore_done=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        # collect_mode config
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)

hopper_onppo_default_config = EasyDict(hopper_onppo_default_config)
main_config = hopper_onppo_default_config

hopper_onppo_create_config = dict(
    env=dict(
        type='mujoco-disc',
        import_names=['dizoo.mujoco.envs.mujoco_env_disc'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
hopper_onppo_create_config = EasyDict(hopper_onppo_create_config)
create_config = hopper_onppo_create_config


def train(args):
    main_config.exp_name = 'data_hopper/onppo_k64_middlenet' + '_seed' + f'{args.seed}'+'_3M'
    serial_pipeline_onpolicy([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(3e6))

if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline_onpolicy

    # for seed in [0,1,2]:
    for seed in [0]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)