from easydict import EasyDict
from ding.rl_utils.mcts.game_base_config import GameBaseConfig, DiscreteSupport

game_config = EasyDict(dict(
    # TODO:
    env_name='PongNoFrameskip-v4',
    image_based=True,
    # TODO
    device='cuda',
    # device='cpu',
    game_wrapper=True,
    action_space_size=6,
    amp_type='none',
    obs_shape=(12, 96, 96),
    image_channel=3,
    gray_scale=False,
    downsample=True,  # Downsample observations before representation network (See paper appendix Network Architecture)

    # debug
    # collector_env_num=1,
    # evaluator_env_num=1,
    # test_max_episode_steps=int(1e2),
    # num_simulations=2,
    # batch_size=8,
    # game_history_max_length=20,
    # total_transitions=int(1e3),
    # num_unroll_steps=5,
    # td_steps=3,

    collector_env_num=8,
    evaluator_env_num=5,
    max_episode_steps=int(1.08e5),
    test_max_episode_steps=int(1.08e5),
    num_simulations=50,
    batch_size=256,
    game_history_max_length=200,
    total_transitions=int(1e5),
    num_unroll_steps=5,
    td_steps=5,

    # TODO
    revisit_policy_search_rate=1,
    # revisit_policy_search_rate=0.99,

    clip_reward=False,
    use_max_priority=True,
    use_priority=True,
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.25,
    auto_td_steps=int(0.3 * 2e5),
    use_root_value=False,  # TODO
    mini_infer_size=2,
    use_augmentation=False,
    vis_result=True,

    priority_prob_alpha=0.6,
    priority_prob_beta=1,  # TODO(pu): 0.4->1
    prioritized_replay_eps=1e-6,

    # UCB formula
    pb_c_base=19652,
    pb_c_init=1.25,

    support_size=300,
    value_support=DiscreteSupport(-300, 300, delta=1),
    reward_support=DiscreteSupport(-300, 300, delta=1),
    max_grad_norm=10,

    max_training_steps=int(1.2e5),  # TODO
    change_temperature=True,
    test_interval=10000,
    log_interval=1000,
    vis_interval=1000,
    checkpoint_interval=100,
    target_model_interval=200,
    save_ckpt_interval=10000,
    discount=0.997,
    dirichlet_alpha=0.3,
    value_delta_max=0.01,
    num_actors=1,
    # network initialization/ & normalization
    episode_life=True,
    init_zero=True,
    state_norm=False,
    # storage efficient
    cvt_string=False,
    # TODO(pu)
    # lr scheduler
    lr_warm_up=0.01,
    lr_init=0.2,
    lr_decay_rate=0.1,
    lr_decay_steps=100000,
    auto_td_steps_ratio=0.3,
    # replay window
    start_transitions=8,
    transition_num=1,
    # frame skip & stack observation
    frame_skip=4,
    stacked_observations=4,
    # coefficient
    reward_loss_coeff=1,
    value_loss_coeff=0.25,
    policy_loss_coeff=1,
    consistency_coeff=2,
    # reward sum
    lstm_hidden_size=512,
    lstm_horizon_len=5,
    # siamese
    proj_hid=1024,
    proj_out=1024,
    pred_hid=512,
    pred_out=1024,

    bn_mt=0.1,
    blocks=1,  # Number of blocks in the ResNet
    channels=64,  # Number of channels in the ResNet
    reduced_channels_reward=16,  # x36 Number of channels in reward head
    reduced_channels_value=16,  # x36 Number of channels in value head
    reduced_channels_policy=16,  # x36 Number of channels in policy head
    resnet_fc_reward_layers=[32],  # Define the hidden layers in the reward head of the dynamic network
    resnet_fc_value_layers=[32],  # Define the hidden layers in the value head of the prediction network
    resnet_fc_policy_layers=[32],  # Define the hidden layers in the policy head of the prediction network
))

game_config = GameBaseConfig(game_config)
