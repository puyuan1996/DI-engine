# from dizoo.mujoco.config.halfcheetah_sac_data_generation_default_config_pu import main_config, create_config
from dizoo.mujoco.config.hopper_sac_data_generation_default_config_pu import main_config, create_config

from ding.entry import serial_pipeline_offline
import os
import torch
from torch.utils.data import DataLoader
from ding.config import read_config, compile_config
from ding.utils.data import create_dataset
import numpy as np

def train(args):
    config = [main_config, create_config]
    input_cfg = config
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=args.seed, auto=True, create_cfg=create_cfg)

    # Dataset
    dataset = create_dataset(cfg)
    print('num_episodes', dataset.__len__())
    # print(dataset.__getitem__(0))
    print(dataset.__getitem__(0)[0]['action'])
    print([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])
    
    episodes_len = np.array([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])
    print('episodes_len',episodes_len)
    index_of_len1000 = np.argwhere(episodes_len==1000).reshape(-1) 
    return_of_len1000 = torch.stack([torch.stack([dataset.__getitem__(episode)[i]['reward'] for i in range(dataset.__getitem__(episode).__len__())],axis=0).sum(0) for episode in list(index_of_len1000)],axis=0)
    print('return_of_len1000',return_of_len1000)
    # stacked action of the first collected episode
    # length 1000, return 3631
    episode_action_8 = torch.stack([dataset.__getitem__(8)[i]['action'] for i in range(dataset.__getitem__(8).__len__())],axis=0)
    episode_reward_8 = torch.stack([dataset.__getitem__(8)[i]['reward'] for i in range(dataset.__getitem__(8).__len__())],axis=0)
    print(episode_reward_8.max(),episode_reward_8.min(),episode_reward_8.mean(),episode_reward_8.std())
    print(episode_action_8.max(0),episode_action_8.min(0),episode_action_8.mean(0),episode_action_8.std(0))

    
    # length 806, return 3030
    episode_action_99 = torch.stack([dataset.__getitem__(99)[i]['action'] for i in range(dataset.__getitem__(99).__len__())],axis=0)
    episode_reward_99 = torch.stack([dataset.__getitem__(99)[i]['reward'] for i in range(dataset.__getitem__(99).__len__())],axis=0)
    print(episode_reward_99.max(),episode_reward_99.min(),episode_reward_99.mean())



    # dataloader = DataLoader(dataset, cfg.policy.learn.batch_size, shuffle=True, collate_fn=lambda x: x)
    # for i, train_data in enumerate(dataloader):
    #     print(i, train_data)
    # serial_pipeline_offline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)