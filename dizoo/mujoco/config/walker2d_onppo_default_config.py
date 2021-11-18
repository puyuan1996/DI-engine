from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

collector_env_num = 1
evaluator_env_num = 1
walker2d_ppo_default_config = dict(
    # exp_name="result_mujoco/wlker2d_onppo_ig",
    exp_name="result_mujoco/wlker2d_onppo_noig",
    env=dict(
        env_id='Walker2d-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        use_act_scale=True,
        n_evaluator_episode=10,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        continuous=True,
        on_policy=True,
        model=dict(
            continuous=True,
            obs_shape=17,
            action_shape=6,
        ),
        learn=dict(
           epoch_per_collect=10,#10,
            update_per_collect=1,
            batch_size=64,#320,
            learning_rate=3e-4,
            value_weight=0.25,#0.5,
            entropy_weight=0,#0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            ignore_done=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            collector_env_num=collector_env_num,
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)
walker2d_ppo_default_config = EasyDict(walker2d_ppo_default_config)
main_config = walker2d_ppo_default_config

walker2d_ppo_create_default_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='ppo', ),
)
walker2d_ppo_create_default_config = EasyDict(walker2d_ppo_create_default_config)
create_config = walker2d_ppo_create_default_config

if __name__ == "__main__":
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
