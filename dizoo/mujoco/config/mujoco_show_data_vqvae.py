# from dizoo.mujoco.config.hopper_dqn_vqvae_data_generation_config import main_config, create_config
from dizoo.mujoco.config.hopper_dqn_data_generation_config import main_config, create_config

from ding.entry import serial_pipeline_offline
import os
import torch
from torch.utils.data import DataLoader
from ding.torch_utils import to_ndarray, to_list, to_tensor
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

    episode0_actions = torch.stack([dataset.__getitem__(0)[i]['action'] for i in range(dataset.__getitem__(0).__len__())],axis=0)
    episode0_rewards = torch.stack([dataset.__getitem__(0)[i]['reward'] for i in range(dataset.__getitem__(0).__len__())],axis=0) 
    episode0_infos_xposition = torch.stack([to_tensor(dataset.__getitem__(0)[i]['info']['x_position']) for i in range(dataset.__getitem__(0).__len__())],axis=0)
    episode0_infos_xvelocity = torch.stack([to_tensor(dataset.__getitem__(0)[i]['info']['x_velocity']) for i in range(dataset.__getitem__(0).__len__())],axis=0)


    episode0_latent_actions = torch.stack([dataset.__getitem__(0)[i]['latent_action'] for i in range(dataset.__getitem__(0).__len__())],axis=0)
    print(episode0_reward.max(),episode0_rewards.min(),episode0_rewards.mean(),episode0_rewards.std())
    print( episode0_actions.max(0), episode0_actions.min(0), episode0_actions.mean(0), episode0_actions.std(0))

    # the num of unique latent actions in each episode
    episodes_num_of_latent_actions = [torch.unique(
                 torch.stack([dataset.__getitem__(episode)[i]['latent_action'] for i in range(dataset.__getitem__(episode).__len__())],axis=0).view(-1)).shape
            for episode in list(index_of_len1000) ]
    
    # the unique latent actions in all episodes
    episodes_unique_latent_actions = torch.unique( torch.stack([
                 torch.stack([dataset.__getitem__(episode)[i]['latent_action'] for i in range(dataset.__getitem__(episode).__len__())], axis=0).view(-1)
            for episode in list(index_of_len1000) ]))

    # np.save('dqn_episode0_infos_xvelocity.npy',episode0_infos_xvelocity)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Hopper-v3 dqn episode0_infos_xvelocity')
    plt.plot(episode0_infos_xvelocity)
    plt.show()
    plt.savefig(f'hopper-v3_dqn_episode0_infos_xvelocity.png')

    # dataloader = DataLoader(dataset, cfg.policy.learn.batch_size, shuffle=True, collate_fn=lambda x: x)
    # for i, train_data in enumerate(dataloader):
    #     print(i, train_data)
    # serial_pipeline_offline(config, seed=args.seed)
    
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title('episode actions')
    # plt.imshow(episode_action)
    # plt.colorbar()
    # plt.show()
    # plt.savefig(f'episode_actions_7400.png')
    # print(f'save episode_actions_7400.png')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)