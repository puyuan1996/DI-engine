import imageio

from dizoo.box2d.lunarlander.config.lunarlander_cont_dqn_vqvae_ensemble_generation_config import main_config, \
    create_config

from ding.entry import collect_episodic_demo_data, eval
import torch
import copy
import torch
from torch.utils.data import DataLoader
from ding.torch_utils import to_ndarray, to_list, to_tensor
from ding.config import read_config, compile_config
from ding.utils.data import create_dataset
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from ding.entry import serial_pipeline_dqn_vqvae_visualize
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle
def train(args):
    # config = [main_config, create_config]
    config = [copy.deepcopy(main_config), copy.deepcopy(create_config)]
    input_cfg = config
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=args.seed, auto=True, create_cfg=create_cfg)

    # for iter in [-1, 0, 40000, 80000, 190000, 380000]:
    for iter in [80000, 190000, 380000]:

    # for iter in [-1]:
        #     if iter == -1:
        #         visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_seed0_3M/action_coverage/iter-best_action/'
        #     else:
        #         visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_seed0_3M/action_coverage/iter-{iter}_action/'

        if iter == -1:
            visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_seed0_3M/position_coverage/iter-best_position/'
        else:
            visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_seed0_3M/position_coverage/iter-{iter}_position/'

        episodes_action = []
        episodes_obs = []

        if not os.path.exists(visualize_path):
                os.mkdir(visualize_path)
        # for seed in range(10):
        for seed in range(0,20):
        # for seed in [0]:

            print(f'iter: {iter}', f'seed: {seed}')

            if iter == -1:
                cfg.policy.collect.data_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_seed0_3M/iter-best_collect_in_' + f'seed{seed}/data_iter-best_1eps.pkl'
            else:
                cfg.policy.collect.data_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_seed0_3M/iter-{iter}_collect_in_' + f'seed{seed}/data_iter-{iter}_1eps.pkl'


            # Dataset
            dataset = create_dataset(cfg)
            # print('num_episodes', dataset.__len__())
            # print('sample action', dataset.__getitem__(0)[0]['action'])
            # print([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])


            episode0_actions_collect_in_seed0 = torch.stack(
                [dataset.__getitem__(0)[i]['action'] for i in range(dataset.__getitem__(0).__len__())], axis=0)

            episode0_rewards_collect_in_seed0 = torch.stack(
                [dataset.__getitem__(0)[i]['reward'] for i in range(dataset.__getitem__(0).__len__())], axis=0)

            episode0_obss_collect_in_seed0 = torch.stack(
                [dataset.__getitem__(0)[i]['obs'] for i in range(dataset.__getitem__(0).__len__())], axis=0)

            print('episode_length:', episode0_actions_collect_in_seed0.shape[0])
            print('episode_return:', episode0_rewards_collect_in_seed0.sum())


            # plot action
            x = episode0_actions_collect_in_seed0[:,0]
            y = episode0_actions_collect_in_seed0[:,1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # sc = ax.scatter(x, y, marker='o')

            # position_mask = np.ma.masked_where((x < 0) & ((y<0.5) & (y>-0.5)), np.arange(episode0_actions_collect_in_seed0.shape[0]))
            # sc = ax.scatter(x, y, marker='o', c=position_mask.mask, cmap='coolwarm')
            # # sc = ax.scatter(x, y, marker='o', c=y, cmap='coolwarm')
            # ax.add_patch(Rectangle(xy=(-1, -0.5), width=1, height=1, linewidth=1,linestyle='dotted', color='red', fill=False))
            #
            # plt.xlabel('Original Action Dim0')
            # plt.ylabel('Original Action Dim1')
            # ax.set_title('Action Coverage')
            # ##fig.colorbar(sc)
            # plt.savefig(f'{visualize_path}' + f'1eps_action_collect_in_seed{seed}_mask.png')

            # plt.show()

            # plot obs
            x = episode0_obss_collect_in_seed0[:,0]
            y = episode0_obss_collect_in_seed0[:,1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sc = ax.scatter(x, y, marker='o', cmap='coolwarm')
            plt.xlim(-1, 1)
            plt.ylim(0, 2)

            plt.xlabel('Horizontal Coordinate x')
            plt.ylabel('Vertical Coordinate y')
            ax.set_title('Position Coverage')
            ##fig.colorbar(sc)
            plt.savefig(f'{visualize_path}' + f'1eps_position_collect_in_seed{seed}.png')


            # episode_actions.append(episode0_actions_collect_in_seed0)
            episodes_obs.append(episode0_obss_collect_in_seed0)

        # episode_actions = torch.cat(episode_actions)
        episodes_obs = torch.cat(episodes_obs)

        # # original action coverage
        x = episodes_obs[:,0]
        y = episodes_obs[:,1]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        sc = ax.scatter(x, y, marker='o', cmap='coolwarm')
        plt.xlim(-1, 1)
        plt.ylim(0, 2)

        plt.xlabel('Horizontal Coordinate x')
        plt.ylabel('Vertical Coordinate y')
        ax.set_title('Position Coverage')
        plt.savefig(f'{visualize_path}' + f'20eps_position.png')

        # # original action coverage
        # x = episode_actions[:,0]
        # y = episode_actions[:,1]
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # sc = ax.scatter(x, y, marker='o')
        #
        # plt.xlabel('Original Action Dim0')
        # plt.ylabel('Original Action Dim1')
        # ax.set_title('Action Coverage')
        # # fig.colorbar(sc)
        # plt.savefig(f'{visualize_path}' + f'20eps_action.png')
        #
        #
        # # K-means action coverage
        # for k in [3,4,8]:
        #     estimator = KMeans(n_clusters=k)  # 构造聚类器
        #     estimator.fit(episode_actions)  # 聚类
        #     labels = estimator.labels_  # 获取聚类标签
        #     cluster_centers = estimator.cluster_centers_  # 获取聚类中心点
        #
        #     x = episode_actions[:,0]
        #     y = episode_actions[:,1]
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     sc = ax.scatter(x, y,  c=labels, marker='o')
        #     sc = ax.scatter(cluster_centers[:,0], cluster_centers[:,1],  c='red', marker='x')
        #
        #     plt.xlabel('Original Action Dim0')
        #     plt.ylabel('Original Action Dim1')
        #     ax.set_title('Action Coverage')
        #     # fig.colorbar(sc)
        #     plt.savefig(f'{visualize_path}' + f'10eps_action_kmeans_{k}.png')




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)
