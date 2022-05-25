# from dizoo.mujoco.config.halfcheetah_sac_data_generation_config_vqvae import main_config, create_config
from dizoo.mujoco.config.hopper_sac_data_generation_config_vqvae import main_config, create_config

from ding.entry import serial_pipeline_offline
import os
import torch
from torch.utils.data import DataLoader
from ding.config import read_config, compile_config
from ding.utils.data import create_dataset
import numpy as np
import matplotlib.pyplot as plt
import gym

def train(args):
    config = [main_config, create_config]
    input_cfg = config
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=args.seed, auto=True, create_cfg=create_cfg)
    cfg.policy.collect.data_path = '/home/puyuan/hopper_td3_seed0/expert_data_1000eps_seed0.pkl'

    # Dataset
    dataset = create_dataset(cfg)
    print('num_episodes', dataset.__len__())
    print(dataset.__getitem__(0)[0]['action'])
    episodes_len = np.array([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])
    print('episodes_len', episodes_len)
    index_of_len1000 = np.argwhere(episodes_len==1000).reshape(-1) 
    print('index_of_len1000', index_of_len1000)
    return_of_len1000 = torch.stack([torch.stack([dataset.__getitem__(episode)[i]['reward'] for i in range(dataset.__getitem__(episode).__len__())],axis=0).sum(0) for episode in list(index_of_len1000)],axis=0)
    print('return_of_len1000', return_of_len1000)


    # episode_actions = torch.stack([dataset.__getitem__(11)[i]['action'] for i in range(dataset.__getitem__(11).__len__())],axis=0)
    # episode_rewards = torch.stack([dataset.__getitem__(11)[i]['reward'] for i in range(dataset.__getitem__(11).__len__())],axis=0) 
    # episode_obss = torch.stack([dataset.__getitem__(11)[i]['obs'] for i in range(dataset.__getitem__(11).__len__())],axis=0) 

    """plot episode rewards"""
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title('Hopper-v3 sac episode0_rewards')
    # plt.plot(episode_rewards)
    # plt.show()
    # plt.savefig(f'hopper-v3_sac_episode0_rewards.png')
    
    """plot episode actions in 3d"""
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib import cm
    # fig = plt.figure(figsize=(16, 12))  #参数为图片大小
    # ax = fig.gca(projection='3d')  # get current axes，且坐标轴是3d的
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("Hopper-v3", alpha=0.5) 
    # # ax.plot(episode_actions[:80,0], episode_actions[:80,1], episode_actions[:80,2], label='hopper-v3_sac_episode_actions')
    # ax.scatter(episode_actions[:80,0], episode_actions[:80,1], episode_actions[:80,2])
    # ax.legend(loc='upper right')
    # plt.show()
    # plt.savefig(f'hopper-v3_sac_episode_actions_0-80_dot.png')

    """plot episode actions in each dim"""
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title('Hopper-v3 sac episode_actions_0dim')
    # plt.plot(episode_actions[:,0])
    # plt.show()
    # plt.savefig(f'hopper-v3_sac_episode_actions_0dim.png')


    """action histogram in each dim"""
    # fig = plt.figure()

    # # Fixing bin edges
    # HIST_BINS = np.linspace(-1, 1, 20)

    # # the histogram of the data
    # n, bins, patches = plt.hist(
    #     episode_actions[:,2].cpu().numpy(), HIST_BINS, density=False, facecolor='g', alpha=0.75
    # )

    # plt.xlabel('actions dim 2')
    # plt.ylabel('Count')
    # plt.title('Histogram of actions dim 2')
    # plt.grid(True)
    # plt.show()
    # plt.savefig(f'hopper-v3_sac_episode_actions_2dim_histogram.png')


    """
    use the disc action to interact with env
    """
    # print('index_of_len1000:', index_of_len1000)
    # for index_of_len1000_ in index_of_len1000:
    #     print(index_of_len1000_)
    #     episode_actions = torch.stack([dataset.__getitem__(index_of_len1000_)[i]['action'] for i in range(dataset.__getitem__(index_of_len1000_).__len__())],axis=0)


    # episode 83: saved return=3617.5715, 
    # episode_actions_numpy return=1091
    # episode_actions_disc: num_bins = 20 return=11, num_bins = 200 return=165, num_bins = 2000 return=285, num_bins = 20000 return=773, num_bins = 200000 return=1091,

    # for index in list(index_of_len1000)[:5]:
    for index in [list(index_of_len1000)[1]]:
        for num_bins in [20,200,2000,20000,200000]:
        # num_bins = 20
            print('num_bins:', num_bins)
            print('#'*20)
            episode_actions = torch.stack([dataset.__getitem__(index)[i]['action'] for i in range(dataset.__getitem__(index).__len__())],axis=0)
            episode_rewards = torch.stack([dataset.__getitem__(index)[i]['reward'] for i in range(dataset.__getitem__(index).__len__())],axis=0)
            # print('episode_rewards', episode_rewards.sum())
            episode_obss = torch.stack([dataset.__getitem__(index)[i]['obs'] for i in range(dataset.__getitem__(index).__len__())],axis=0) 

            # num_bins = 200
            HIST_BINS = np.linspace(-1, 1, num_bins)
            episode_actions_numpy = episode_actions.cpu().numpy()
            episode_actions_dim0_ind = np.digitize(episode_actions_numpy[:,0], HIST_BINS)
            episode_actions_dim1_ind = np.digitize(episode_actions_numpy[:,1], HIST_BINS)
            episode_actions_dim2_ind = np.digitize(episode_actions_numpy[:,2], HIST_BINS)

            episode_actions_ind = np.stack([episode_actions_dim0_ind,episode_actions_dim1_ind,episode_actions_dim2_ind]).transpose()
            episode_actions_disc =   -1 + 2/num_bins * (episode_actions_ind-1)

            """8 envs, using env manager"""
            # from functools import partial
            # from ding.envs import create_env_manager, get_vec_env_setting
            # env_setting = None
            # env_fn = None if env_setting is None else env_setting[0]
            # if env_setting is None:
            #     env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
            # else:
            #     env_fn, _, evaluator_env_cfg = env_setting
            # evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
            # evaluator_env.seed(args.seed, dynamic_seed=False)
            # env = evaluator_env
            # env.launch()
            
            """one env, not using env manager"""
            env = gym.make('Hopper-v3')
            args.seed=0
            env.seed(args.seed)
            # env.seed(args.seed, dynamic_seed=False)
            from ding.utils import set_pkg_seed
            set_pkg_seed(args.seed, use_cuda=cfg.policy.cuda)

            total_return = []
            total_length = []
            episode_cnt = 1
            for eps_id in range(episode_cnt):
                """one env, not using env manager"""
                observation = env.reset()
                
                """8 envs, using env manager"""
                # env.reset()
                # obs = env.ready_obs

                done = False
                eps_length = 0
                eps_return = 0
                while not done:
                    # action = env.action_space.sample()
                    action = episode_actions_disc[eps_length]
                    # action = episode_actions_numpy[eps_length]

                    """8 envs, using env manager"""
                    # action = {i:action for i in range(8)}
                    # timesteps = env.step(action)
                    # observation, reward, done, info = timesteps[0]

                    """one env, not using env manager"""
                    observation, reward, done, info = env.step(action)
                    # env.render()
                    eps_return += reward
                    eps_length += 1
                    if done:
                        # print(observation.shape)
                        total_return.append(eps_return)
                        total_length.append(eps_length)
                        # print("observation:", observation, "reward:", reward, "done:", done, "info:", info, )
                        print("eps: {} done. ".format(eps_id), "eps_length:", eps_length, "eps_return:", eps_return,
                            "final_reward:",
                            reward, "info:", info, )
                        break

            # print("return mean:", np.mean(total_return), 'std:', np.std(total_return), 'max:', np.max(total_return), 'min:',
            #     np.min(total_return))
            # print("length mean:", np.mean(total_length), 'std:', np.std(total_length), 'max:', np.max(total_length), 'min:',
            #     np.min(total_length))
            env.close()

    # episodes_len = np.array([len(dataset.__getitem__(i)) for i in range(dataset.__len__())])
    # print('episodes_len',episodes_len)
    # index_of_len1000 = np.argwhere(episodes_len==1000).reshape(-1) 
    # return_of_len1000 = torch.stack([torch.stack([dataset.__getitem__(episode)[i]['reward'] for i in range(dataset.__getitem__(episode).__len__())],axis=0).sum(0) for episode in list(index_of_len1000)],axis=0)
    # print('return_of_len1000',return_of_len1000)
    # # stacked action of the first collected episode
    # # length 1000, return 3631
    # episode_action_8 = torch.stack([dataset.__getitem__(8)[i]['action'] for i in range(dataset.__getitem__(8).__len__())],axis=0)
    # episode_reward_8 = torch.stack([dataset.__getitem__(8)[i]['reward'] for i in range(dataset.__getitem__(8).__len__())],axis=0)
    # print(episode_reward_8.max(),episode_reward_8.min(),episode_reward_8.mean(),episode_reward_8.std())
    # print(episode_action_8.max(0),episode_action_8.min(0),episode_action_8.mean(0),episode_action_8.std(0))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)