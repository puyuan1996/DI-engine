import imageio

from dizoo.mujoco.config.hopper_sac_config import main_config, \
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
from sklearn import manifold, datasets

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from ding.entry import serial_pipeline_dqn_vqvae_visualize
from sklearn.cluster import KMeans


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
    episode_actions_tsne_el1000_erlt3000 = []
    episode_actions_tsne_el1000_erlt3500 = []

    el1000_erlt3000_cnt = 0
    el1000_erlt3500_cnt = 0


    # for iter in [-1, 0, 100000, 200000, 1000000]:
    for iter in [ 500000]:
    # for iter in [1000000]:

        if iter == -1:
            visualize_path = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/action_coverage/iter-best_action'
        else:
            visualize_path = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/action_coverage/iter-{iter}_action'

        # visualize_path = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/action_coverage/el1000_erlt3000'

        if not os.path.exists(visualize_path):
                os.mkdir(visualize_path)
        episode_actions_tsne = []

        # for seed in range(10):
        for seed in [8]:

            # for seed in range(10,20):

            print(f'iter: {iter}', f'seed: {seed}')

            if iter == -1:
                cfg.policy.collect.data_path = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/iter-best_collect_in_' + f'seed{seed}/data_iter-best_1eps.pkl'
            else:
                cfg.policy.collect.data_path = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/iter-{iter}_collect_in_' + f'seed{seed}/data_iter-{iter}_1eps.pkl'

            # Dataset
            cfg.policy.collect.data_type = 'naive'
            dataset = create_dataset(cfg)
            # print('num_episodes', dataset.__len__())
            # print('sample action', dataset.__getitem__(0)[0]['action'])
            # print([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])

            episode_actions_collect_in_one_seed = torch.stack(
                [dataset.__getitem__(0)[i]['action'] for i in range(dataset.__getitem__(0).__len__())], axis=0)

            episode0_rewards_collect_in_seed0 = torch.stack(
                [dataset.__getitem__(0)[i]['reward'] for i in range(dataset.__getitem__(0).__len__())], axis=0)

            print('episode_length:', episode_actions_collect_in_one_seed.shape[0])
            print('episode_return:', episode0_rewards_collect_in_seed0.sum())

            """t-SNE"""
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(episode_actions_collect_in_one_seed)

            # print("Org data dimension is {}, Embedded data dimension is {}".format(
            #     episode_actions_collect_in_one_seed.shape[-1], X_tsne.shape[-1]))

            """嵌入空间可视化"""
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化[0,1]

            x = X_norm[:, 0]
            y = X_norm[:, 1]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # sc = ax.scatter(x, y, marker='o')
            sc = ax.scatter(x, y, marker='o', cmap='coolwarm')
            plt.xlabel('t-SNE Dim0')
            plt.ylabel('t-SNE Dim1')
            ax.set_title('Action Coverage')
            ##fig.colorbar(sc)
            plt.savefig(f'{visualize_path}' + f'/1eps_action_collect_in_seed{seed}_arq.png')
            #
            #
            # episode_actions_tsne.append(X_norm)

            # if episode_actions_collect_in_one_seed.shape[0]==1000 and episode0_rewards_collect_in_seed0.sum()>3000:
            #     # episode_length=1000 and episode_return>3000
            #     el1000_erlt3000_cnt +=1
            #     print(f'iter: {iter}', f'seed: {seed}', 'episode_length=1000 and episode_return>3000')
            #     print(f'el1000_erlt3000_cnt: {el1000_erlt3000_cnt}')
            #     episode_actions_tsne_el1000_erlt3000.append(X_norm)
            #
            # if episode_actions_collect_in_one_seed.shape[0] == 1000 and episode0_rewards_collect_in_seed0.sum() > 3500:
            #     # episode_length=1000 and episode_return>3500
            #     el1000_erlt3500_cnt += 1
            #     print(f'iter: {iter}', f'seed: {seed}', 'episode_length=1000 and episode_return>3500')
            #     print(f'el1000_erlt3500_cnt: {el1000_erlt3500_cnt}')
            #     episode_actions_tsne_el1000_erlt3500.append(X_norm)

        # episode_actions_tsne = np.concatenate(episode_actions_tsne)
        #
        # # # original action coverage: tsne
        # x = episode_actions_tsne[:, 0]
        # y = episode_actions_tsne[:, 1]
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # sc = ax.scatter(x, y, marker='o')
        #
        # # plt.xlabel('Original Action Dim0')
        # # plt.ylabel('Original Action Dim1')
        # ax.set_title('Action Coverage')
        # # fig.colorbar(sc)
        # plt.savefig(f'{visualize_path}' + f'/10eps_action.png')

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

    # episode_actions_tsne_el1000_erlt3000 = np.concatenate(episode_actions_tsne_el1000_erlt3000)
    # # # original action coverage: tsne
    # x = episode_actions_tsne_el1000_erlt3000[:, 0]
    # y = episode_actions_tsne_el1000_erlt3000[:, 1]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # sc = ax.scatter(x, y, marker='o')
    # sc = ax.scatter(x, y, marker='o', cmap='coolwarm')
    #
    # # plt.xlabel('Original Action Dim0')
    # # plt.ylabel('Original Action Dim1')
    # ax.set_title('Action Coverage')
    # # fig.colorbar(sc)
    # plt.savefig(f'{visualize_path}' + f'/el1000_erlt3000_action.png')
    #
    # episode_actions_tsne_el1000_erlt3500 = np.concatenate(episode_actions_tsne_el1000_erlt3500)
    # # # original action coverage: tsne
    # x = episode_actions_tsne_el1000_erlt3500[:, 0]
    # y = episode_actions_tsne_el1000_erlt3500[:, 1]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # sc = ax.scatter(x, y, marker='o')
    #
    # # plt.xlabel('Original Action Dim0')
    # # plt.ylabel('Original Action Dim1')
    # ax.set_title('Action Coverage')
    # # fig.colorbar(sc)
    # plt.savefig(f'{visualize_path}' + f'/el1000_erlt3500_action.png')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)
