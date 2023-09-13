from easydict import EasyDict

toy_ppo_config = dict(
    exp_name='toy_ppo_igdone-true_ew0001_denserew_seed0',
    env=dict(
        collector_env_num=10,
        evaluator_env_num=5,
        act_transform=True,
        n_evaluator_episode=5,
        stop_value=1,
    ),
    policy=dict(
        cuda=False,
        action_space='continuous',
        recompute_adv=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64],
            action_space='continuous',
            actor_head_layer_num=0,
            critic_head_layer_num=0,
            sigma_type='conditioned',
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            # entropy_weight=0.,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            # ignore_done=False,
            ignore_done=True,

        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=500, ))
    ),
)
toy_ppo_config = EasyDict(toy_ppo_config)
main_config = toy_ppo_config
toy_ppo_create_config = dict(
    env=dict(
        type='toy_env',
        import_names=['dizoo.toy.envs.toy_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
toy_ppo_create_config = EasyDict(toy_ppo_create_config)
create_config = toy_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c toy_ppo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
