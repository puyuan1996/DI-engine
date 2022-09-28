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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
module_path = os.path.dirname(__file__)

from matplotlib.ticker import FuncFormatter
def y_update_scale_value(temp, position):
    # result = temp//1e+6
    # return "{}M".format(int(result))
    if temp/1e+6 % 1 == 0:
        result = int(temp//1e+6)
    else:
        result = temp/1e+6
    return "{}M".format(result)

sns.set()
windowsize=10
f = plt.figure(figsize=(7, 5.5))
size1=23
size2=16
size3=2.5
size4=12

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

    iter_noop_action_cnt_list  = []
    iter_total_action_cnt_list = []
    iter_total_fuel_cnt_list = []
    iter_x_fuel_cnt_list = []
    iter_y_fuel_cnt_list = []



    # for iter in [-1, 0, 10000, 20000, 60000, 120000, 180000]:
    # for iter in [0, 10000, 20000, 60000, 120000, -1]:
    for iter in [-1]:
        if iter == -1:
            visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/sac_seed0_3M/action_coverage/iter-best_action/'
        else:
            visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/sac_seed0_3M/action_coverage/iter-{iter}_action/'

        episode_actions = []

        iter_noop_action_cnt = 0
        iter_total_action_cnt = 0
        iter_total_fuel_cnt = 0
        iter_x_fuel_cnt = 0
        iter_y_fuel_cnt = 0


        # for seed in range(10):
        # for seed in range(0,20):
        for seed in [8]:

            print(f'iter: {iter}', f'seed: {seed}')

            if iter == -1:
                cfg.policy.collect.data_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/sac_seed0_3M/iter-best_collect_in_' + f'seed{seed}/data_iter-best_1eps.pkl'
            else:
                cfg.policy.collect.data_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/sac_seed0_3M/iter-{iter}_collect_in_' + f'seed{seed}/data_iter-{iter}_1eps.pkl'


            # Dataset
            dataset = create_dataset(cfg)
            # print('num_episodes', dataset.__len__())
            # print('sample action', dataset.__getitem__(0)[0]['action'])
            # print([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])


            episode0_actions_collect_in_seed0 = torch.stack(
                [dataset.__getitem__(0)[i]['action'] for i in range(dataset.__getitem__(0).__len__())], axis=0)

            episode0_rewards_collect_in_seed0 = torch.stack(
                [dataset.__getitem__(0)[i]['reward'] for i in range(dataset.__getitem__(0).__len__())], axis=0)

            print('episode_length:',episode0_actions_collect_in_seed0.shape[0])
            print('episode_return:',episode0_rewards_collect_in_seed0.sum())

            iter_total_action_cnt += episode0_actions_collect_in_seed0.shape[0]

            x = episode0_actions_collect_in_seed0[:,0]
            y = episode0_actions_collect_in_seed0[:,1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # sc = ax.scatter(x, y, marker='o')

            position_mask = np.ma.masked_where((x < 0) & ((y<0.5) & (y>-0.5)), np.arange(episode0_actions_collect_in_seed0.shape[0]))

            iter_x_fuel_cnt += (x * (x > 0)).sum()
            # iter_y_fuel_cnt +=  (y*((y>0.5) | (y<-0.5))).sum()
            iter_y_fuel_cnt += (y * (y > 0.5) ).sum() + (abs( y * (y < -0.5)) ).sum()

            iter_total_fuel_cnt += (x*(x>0)).sum() + (y * (y > 0.5) ).sum() + (abs( y * (y < -0.5)) ).sum()

            sc = ax.scatter(x, y, marker='o', c=position_mask.mask, cmap='coolwarm')
            # sc = ax.scatter(x, y, marker='o', c=y, cmap='coolwarm')
            ax.add_patch(Rectangle(xy=(-1, -0.5), width=1, height=1, linewidth=1,linestyle='dotted', color='red', fill=False))

            # plt.xlabel('Original Action Dim0')
            # plt.ylabel('Original Action Dim1')
            # ax.set_title('Action Coverage')
            # ##fig.colorbar(sc)
            # plt.savefig(f'{visualize_path}' + f'1eps_action_collect_in_seed{seed}_mask.png')

            # plt.gca().xaxis.set_major_formatter(FuncFormatter(y_update_scale_value))
            plt.tick_params(axis='both', labelsize=size2)
            plt.legend(loc="best", fontsize=size4)  # 显示图例
            plt.xlabel('Original Action Dim0', fontsize=size1)
            plt.ylabel('Original Action Dim1', fontsize=size1)
            plt.tight_layout()
            # f.savefig('./Halfcheetah.pdf', bbox_inches='tight')
            plt.savefig(f'{visualize_path}' + f'1eps_action_collect_in_seed{seed}_mask_1.png')






        # episode_actions = torch.cat(episode_actions)

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
