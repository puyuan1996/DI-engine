from copy import deepcopy
import os
import torch

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

from ding.entry import serial_pipeline_bc, collect_episodic_demo_data, episode_to_transitions_filter, eval
from ding.config import read_config, compile_config
from ding.policy import create_policy
from dizoo.gfootball.entry.gfootball_il_config import gfootball_il_main_config, gfootball_il_create_config
from dizoo.gfootball.model.q_network.football_q_network import FootballNaiveQ
from dizoo.gfootball.model.bots.rule_based_bot_model import FootballRuleBaseModel


# in gfootball env: 3000 transitions = one episode
# 3e5 transitions = 200 episode, The memory needs about 350G
seed = 0
gfootball_il_main_config.exp_name = 'data_gfootball/gfootball_easy_il_rule_200ep_lt0_seed0_lsce_cecw_wd1e-4'
demo_episodes = 200  # key hyper-parameter
data_path_episode = dir_path + f'/gfootball_easy_rule_{demo_episodes}eps.pkl'
data_path_transitions_lt0 = dir_path + f'/gfootball_easy_rule_{demo_episodes}eps_transitions_lt0.pkl'

"""
phase 1: train/obtain expert policy
"""
input_cfg = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
if isinstance(input_cfg, str):
    cfg, create_cfg = read_config(input_cfg)
else:
    cfg, create_cfg = input_cfg
create_cfg.policy.type = create_cfg.policy.type + '_command'
env_fn = None
cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)

football_rule_base_model = FootballRuleBaseModel()
expert_policy = create_policy(cfg.policy, model=football_rule_base_model,
                              enable_field=['learn', 'collect', 'eval', 'command'])

# collect expert demo data
state_dict = expert_policy.collect_mode.state_dict()
collect_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
eval_config = deepcopy(collect_config)

# eval demo model
# if save replay
# eval(eval_config, seed=seed, model=football_rule_base_model, replay_path=dir_path + f'/gfootball_rule_replay/')
# if not save replay
# eval(eval_config, seed=seed, model=football_rule_base_model, state_dict=state_dict)

# collect demo data
collect_episodic_demo_data(
    collect_config, seed=seed, expert_data_path=data_path_episode, collect_count=demo_episodes,
    model=football_rule_base_model, state_dict=state_dict
)
# only use the episode whose return is larger than 0 as demo data
episode_to_transitions_filter(data_path=data_path_episode, expert_data_path=data_path_transitions_lt0, nstep=1,
                              min_episode_return=1)

"""
phase 2: il training
"""
il_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
il_config[0].policy.learn.train_epoch = 1000  # key hyper-parameter

il_config[0].env.stop_value = 999  # Don't stop until training <train_epoch> epochs
il_config[0].policy.eval.evaluator.multi_gpu = False
football_naive_q = FootballNaiveQ()

il_config[0].policy.learn.show_accuracy = False
il_config[0].policy.learn.ce_class_weight = True
il_config[0].policy.learn.lsce = True

_, converge_stop_flag = serial_pipeline_bc(il_config, seed=seed, data_path=data_path_transitions_lt0,
                                           model=football_naive_q)

if il_config[0].policy.test_accuracy:
    """
    phase 3: test accuracy in train dataset and validation dataset
    """
    il_model_path = il_config[0].policy.il_model_path

    # load trained model
    il_config[0].policy.learn.batch_size = int(3000)  # the total dataset
    il_config[0].policy.learn.train_epoch = 1
    il_config[0].policy.learn.show_accuracy = True
    state_dict = torch.load(il_model_path, map_location='cpu')
    football_naive_q.load_state_dict(state_dict['model'])

    # calculate accuracy in train dataset
    data_path_transitions_lt0 = dir_path + f'/gfootball_rule_100eps_transitions_lt0.pkl'
    print('==' * 10)
    print('calculate accuracy in train dataset' * 10)
    print('==' * 10)
    _, converge_stop_flag = serial_pipeline_bc(il_config, seed=seed, data_path=data_path_transitions_lt0,
                                               model=football_naive_q)

    # calculate accuracy in validation dataset
    data_path_transitions_lt0_test = dir_path + f'/gfootball_rule_50eps_transitions_lt0.pkl'
    print('==' * 10)
    print('calculate accuracy in validation dataset' * 10)
    print('==' * 10)
    _, converge_stop_flag = serial_pipeline_bc(il_config, seed=seed, data_path=data_path_transitions_lt0_test,
                                               model=football_naive_q)
