from dizoo.box2d.lunarlander.config.lunarlander_cont_dqn_vqvae_ensemble_generation_config import main_config, create_config

from ding.entry import collect_episodic_demo_data, eval
import torch
import copy
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

    cfg.policy.collect.data_path =  '/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_crlw1_seed1_3M/collect_in_seed1_q_value_mapping/data_iteration_best_1eps.pkl'

    # Dataset
    dataset = create_dataset(cfg)
    print('num_episodes', dataset.__len__())
    # print(dataset.__getitem__(0))
    print(dataset.__getitem__(0)[0]['action'])
    print([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])

    episodes_len = np.array([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])
    print('episodes_len', episodes_len)
    index_of_len1000 = np.argwhere(episodes_len == 1000).reshape(-1)
    return_of_len1000 = torch.stack([torch.stack(
        [dataset.__getitem__(episode)[i]['reward'] for i in range(dataset.__getitem__(episode).__len__())], axis=0).sum(
        0) for episode in list(index_of_len1000)], axis=0)
    print('return_of_len1000', return_of_len1000)
    # stacked action of the first collected episode

    episode0_actions = torch.stack(
        [dataset.__getitem__(0)[i]['action'] for i in range(dataset.__getitem__(0).__len__())], axis=0)
    episode0_rewards = torch.stack(
        [dataset.__getitem__(0)[i]['reward'] for i in range(dataset.__getitem__(0).__len__())], axis=0)
    episode0_infos_xposition = torch.stack(
        [to_tensor(dataset.__getitem__(0)[i]['info']['x_position']) for i in range(dataset.__getitem__(0).__len__())],
        axis=0)
    episode0_infos_xvelocity = torch.stack(
        [to_tensor(dataset.__getitem__(0)[i]['info']['x_velocity']) for i in range(dataset.__getitem__(0).__len__())],
        axis=0)

    episode0_latent_actions = torch.stack(
        [dataset.__getitem__(0)[i]['latent_action'] for i in range(dataset.__getitem__(0).__len__())], axis=0)
    print(episode0_rewards.max(), episode0_rewards.min(), episode0_rewards.mean(), episode0_rewards.std())
    print(episode0_actions.max(0), episode0_actions.min(0), episode0_actions.mean(0), episode0_actions.std(0))

    # the num of unique latent actions in each episode
    episodes_num_of_latent_actions = [torch.unique(
        torch.stack(
            [dataset.__getitem__(episode)[i]['latent_action'] for i in range(dataset.__getitem__(episode).__len__())],
            axis=0).view(-1)).shape
                                      for episode in list(index_of_len1000)]

    # the unique latent actions in all episodes
    episodes_unique_latent_actions = torch.unique(torch.stack([
        torch.stack(
            [dataset.__getitem__(episode)[i]['latent_action'] for i in range(dataset.__getitem__(episode).__len__())],
            axis=0).view(-1)
        for episode in list(index_of_len1000)]))

    # np.save('dqn_episode0_infos_xvelocity.npy',episode0_infos_xvelocity)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_title('lunarlander dqn episode0_infos_xvelocity')
    # plt.plot(episode0_infos_xvelocity)
    # plt.show()
    # plt.savefig(f'lunarlander_dqn_episode0_infos_xvelocity.png')
    ax.set_title('lunarlander dqn episode0_rewards')
    plt.plot(episode0_rewards)
    plt.show()
    plt.savefig(f'lunarlander_dqn_episode0_rewards.png')



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)