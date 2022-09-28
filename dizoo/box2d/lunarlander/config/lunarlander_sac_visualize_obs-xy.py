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

# sns.set()
windowsize=10
# f = plt.figure(figsize=(7, 5.5))
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

    for iter in [-1, 10000, 20000, 60000, 120000]:
    # for iter in [20000]:
    # for iter in [-1]:
        #     if iter == -1:
        #         visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/sac_seed0_3M/action_coverage/iter-best_action/'
        #     else:
        #         visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/sac_seed0_3M/action_coverage/iter-{iter}_action/'

        if iter == -1:
            visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/sac_seed0_3M/position_coverage/iter-best_position/'
        else:
            visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/sac_seed0_3M/position_coverage/iter-{iter}_position/'

        episodes_action = []
        episodes_obs = []

        if not os.path.exists(visualize_path):
                os.mkdir(visualize_path)
        # for seed in range(10):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for seed in range(0,20):
        # for seed in [0]:

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

            episode0_obss_collect_in_seed0 = torch.stack(
                [dataset.__getitem__(0)[i]['obs'] for i in range(dataset.__getitem__(0).__len__())], axis=0)

            print('episode_length:', episode0_actions_collect_in_seed0.shape[0])
            print('episode_return:', episode0_rewards_collect_in_seed0.sum())


            # plot action
            x = episode0_actions_collect_in_seed0[:,0]
            y = episode0_actions_collect_in_seed0[:,1]
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # sc = ax.scatter(x, y, marker='o')

            # plot obs
            x = episode0_obss_collect_in_seed0[:,0]
            y = episode0_obss_collect_in_seed0[:,1]

            """
            return color
            - [, -100]
            - [ -100, 0]
            - [ 0,100]
            - [100,200]
            - [200,]
            """
            if episode0_rewards_collect_in_seed0.sum()<-100:
                color = '#e2fdff'
            elif episode0_rewards_collect_in_seed0.sum()<0:
                color = '#bfd7ff'
            elif episode0_rewards_collect_in_seed0.sum()<100:
                color = '#9bb1ff'
            elif episode0_rewards_collect_in_seed0.sum()<200:
                color = '#788bff'
            else:
                color = '#5465ff'

            # sc = ax.scatter(x, y, marker='o', cmap='coolwarm')
            sc = ax.scatter(x, y, marker='.', color=color) #, alpha=0.1, s=0.1)

            plt.xlim(-1, 1)
            plt.ylim(0, 2)

            plt.xlabel('Horizontal Coordinate x')
            plt.ylabel('Vertical Coordinate y')
            ax.set_title('Position Coverage')
            ##fig.colorbar(sc)
            plt.savefig(f'{visualize_path}' + f'1eps_position_collect_in_seed{seed}_1.png')

            # plt.gca().xaxis.set_major_formatter(FuncFormatter(y_update_scale_value))
            # plt.tick_params(axis='both', labelsize=size2)
            plt.legend(loc="best", fontsize=size4)  # 显示图例
            plt.xlabel('Horizontal Coordinate x', fontsize=size1)
            plt.ylabel('Vertical Coordinate y', fontsize=size1)
            plt.tight_layout()
            plt.savefig(f'{visualize_path}' + f'1eps_position_collect_in_seed{seed}_1.png')

            # episode_actions.append(episode0_actions_collect_in_seed0)
            episodes_obs.append(episode0_obss_collect_in_seed0)

        # episode_actions = torch.cat(episode_actions)
        episodes_obs = torch.cat(episodes_obs)

        # # original action coverage
        x = episodes_obs[:,0]
        y = episodes_obs[:,1]

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # sc = ax.scatter(x, y, marker='o', cmap='coolwarm')
        # plt.xlim(-1, 1)
        # plt.ylim(0, 2)

        # plt.xlabel('Horizontal Coordinate x')
        # plt.ylabel('Vertical Coordinate y')
        # ax.set_title('Position Coverage')
        # plt.savefig(f'{visualize_path}' + f'20eps_position.png')

        # plt.gca().xaxis.set_major_formatter(FuncFormatter(y_update_scale_value))
        # plt.tick_params(axis='both', labelsize=size2)
        plt.legend(loc="best", fontsize=size4)  # 显示图例
        plt.xlabel('Horizontal Coordinate x', fontsize=size1)
        plt.ylabel('Vertical Coordinate y', fontsize=size1)
        plt.tight_layout()
        # f.savefig('./Halfcheetah.pdf', bbox_inches='tight')
        plt.savefig(f'{visualize_path}' + f'20eps_position_1.png')







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)
