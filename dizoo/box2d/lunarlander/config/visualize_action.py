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
    visualize_path = f'/Users/puyuan/code/DI-engine/data_lunarlander_visualize/'

    # 3000000 menas iter_best
    env_steps = [0, 100000, 200000, 500000, 1000000, 3000000]

    iter_noop_action_ratio_list_sac = [0.20149253731343283, 0.13633004926108375, 0.18965153115100317,
                                       0.22236331042724827, 0.30537061695517087, 0.3254766949152542]
    iter_noop_action_ratio_list_dqn_cluster_k8 = [0.0, 0.020988407565588774, 0.10272410896435857, 0.17611070648215588,
                                                  0.2738627889634601, 0.2373341375150784]
    iter_noop_action_ratio_list_adq_k8 = [1.0, 0.05225, 0.017078577879957303, 0.15924617196702, 0.07615078613373745,
                                          0.11682385153635534]

    # 折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(env_steps, iter_noop_action_ratio_list_sac, 's-', color='r', label="SAC")  # s-:方形
    # ax.plot(env_steps, iter_noop_action_ratio_list_sac, 's-', color='r', label="TD3")  # s-:方形
    # ax.plot(env_steps, iter_noop_action_ratio_list_dqn_cluster_k8, 's-', color='g', label="DQN-cluster-K8")  # s-:方形
    # ax.plot(env_steps, iter_noop_action_ratio_list_adq_k8, 's-', color='b', label="ADQ-K8")  # s-:方形

    plt.plot(env_steps, iter_noop_action_ratio_list_sac, 's-', label="TD3")  # s-:方形
    plt.plot(env_steps, iter_noop_action_ratio_list_dqn_cluster_k8, 's-', label="DQN-cluster-K8")  # s-:方形
    plt.plot(env_steps, iter_noop_action_ratio_list_adq_k8, 's-', label="ADQ-K8")  # s-:方形

    # plt.xlabel("Env Steps")  # 横坐标名字
    # plt.ylabel("Noop Action Ratio")  # 纵坐标名字
    # plt.legend(loc="best")  # 图例
    # plt.savefig(f'{visualize_path}' + f'20eps_statistics_Noop_Action_Ratio.png')


    plt.gca().xaxis.set_major_formatter(FuncFormatter(y_update_scale_value))
    plt.tick_params(axis='both', labelsize=size2)
    # plt.title('Halfcheetah', fontsize=size1)
    # plt.legend(loc='lower right', fontsize=size4)  # 显示图例
    plt.legend(loc="best", fontsize=size4)  # 显示图例
    plt.xlabel('Env Steps', fontsize=size1)
    plt.ylabel('Noop Action Ratio', fontsize=size1)
    plt.tight_layout()
    # f.savefig('./Halfcheetah.pdf', bbox_inches='tight')
    plt.savefig(f'{visualize_path}' + f'20eps_statistics_Noop_Action_Ratio_1.png')


    iter_total_fuel_cnt_list_sac = [1477.6442, 3485.1467, 5556.3379, 7884.2295, 3752.4250, 2335.7441]
    iter_total_fuel_cnt_list_dqn_cluster_k8 = [1960.0189, 8042.9780, 4356.8228, 3853.6658, 3336.0142, 2004.0413]
    iter_total_fuel_cnt_list_adq_k8 = [0., 18082.0176, 9974.0625, 5523.7437, 3842.2612, 2791.0762]

    # 折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = ax.twinx()
    # ax.plot(env_steps, iter_total_fuel_cnt_list_sac, 'o-', color='r', label="SAC")  # o-:圆形
    ax.plot(env_steps, iter_total_fuel_cnt_list_sac, 'o-', color='r', label="TD3")  # o-:圆形
    ax.plot(env_steps, iter_total_fuel_cnt_list_dqn_cluster_k8, 'o-', color='g', label="DQN-cluster-K8")  # o-:圆形
    ax.plot(env_steps, iter_total_fuel_cnt_list_adq_k8, 'o-', color='b', label="ADQ-K8")  # o-:圆形

    plt.xlabel("Env Steps")  # 横坐标名字
    plt.ylabel("Average Total Fuel Count")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.savefig(f'{visualize_path}' + f'20eps_statistics_Total_Fuel_Count.png')

    # print('iter_noop_action_ratio_list',  iter_noop_action_ratio_list)
    # print('iter_total_action_cnt_list',  iter_total_action_cnt_list)
    # print('iter_noop_action_cnt_list',  iter_noop_action_ratio_list)
    # print('iter_total_fuel_cnt_list',  iter_total_fuel_cnt_list)
    # print('iter_x_fuel_cnt_list',  iter_x_fuel_cnt_list)
    # print('iter_y_fuel_cnt_list',  iter_y_fuel_cnt_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)
