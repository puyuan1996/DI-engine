from dizoo.mujoco.config.halfcheetah_dqn_vqvae_data_generation_config import main_config, create_config
from ding.entry import serial_pipeline_offline
import os
import torch
from torch.utils.data import DataLoader
from ding.config import read_config, compile_config
from ding.utils.data import create_dataset

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
    print(dataset.__len__())
    # print(dataset.__getitem__(0))
    print(dataset.__getitem__(0)[0]['action'])

    # episode_action = []
    # for i in range(dataset.__getitem__(0).__len__()):  # length of the firse collected episode
    #     episode_action.append(dataset.__getitem__(0)[i]['action'])
    
    # stacked action of the first collected episode
    episode_action = torch.stack([dataset.__getitem__(0)[i]['action'] for i in range(dataset.__getitem__(0).__len__())],axis=0)

    # dataloader = DataLoader(dataset, cfg.policy.learn.batch_size, shuffle=True, collate_fn=lambda x: x)
    # for i, train_data in enumerate(dataloader):
    #     print(i, train_data)
    # serial_pipeline_offline(config, seed=args.seed)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('episode actions')
    plt.imshow(episode_action)
    plt.colorbar()
    plt.show()
    plt.savefig(f'episode_actions_7400.png')
    print(f'save episode_actions_7400.png')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)