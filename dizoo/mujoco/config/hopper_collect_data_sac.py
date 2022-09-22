from dizoo.mujoco.config.hopper_sac_config import main_config, \
    create_config

from ding.entry import collect_episodic_demo_data, eval
import torch
import copy
import os


def eval_ckpt(args):
    config = copy.deepcopy([main_config, create_config])
    eval(config, seed=args.seed, load_path=main_config.policy.model_path)
    # eval(config, seed=args.seed, load_path=main_config.policy.model_path, replay_path='/Users/puyuan/code/DI-engine/data_hopper_visualize/')


# TODO(pu): config
#  model_path, replay_path, save_replay_gif=True,
#  seed
def generate(args):
    config = copy.deepcopy([main_config, create_config])
    config[0].env.collector_env_num=1
    config[0].env.evaluator_env_num=1
    config[0].env.n_evaluator_episode=1
    if args.iter == -1:
        config[0].policy.model_path = '/Users/puyuan/code/DI-engine/data_hopper/sac_seed0_1M/ckpt/ckpt_best.pth.tar'
        config[0].env.replay_path = '/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/iter-best_collect_in_' + f'seed{args.seed}'
        config[0].env.save_replay_gif = True
        expert_data_path_prefix = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/iter-best_collect_in_' + f'seed{args.seed}'
        if not os.path.exists(expert_data_path_prefix):
            os.mkdir(expert_data_path_prefix)
        expert_data_path = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/iter-best_collect_in_' + f'seed{args.seed}/data_iter-best_1eps.pkl'
    else:
        config[0].policy.model_path = f'/Users/puyuan/code/DI-engine/data_hopper/sac_seed0_1M/ckpt/iteration_{args.iter}.pth.tar'
        config[0].env.replay_path = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/iter-{args.iter}_collect_in_' + f'seed{args.seed}'
        config[0].env.save_replay_gif = True
        expert_data_path_prefix = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/iter-{args.iter}_collect_in_' + f'seed{args.seed}'
        if not os.path.exists(expert_data_path_prefix):
            os.mkdir(expert_data_path_prefix)
        expert_data_path = f'/Users/puyuan/code/DI-engine/data_hopper_visualize/sac_seed0_1M/iter-{args.iter}_collect_in_' + f'seed{args.seed}/data_iter-{args.iter}_1eps.pkl'

    state_dict = torch.load(config[0].policy.model_path, map_location='cpu')
    collect_episodic_demo_data(
        config,
        collect_count=1,
        seed=args.seed,
        expert_data_path=expert_data_path,
        state_dict=state_dict
    )


if __name__ == "__main__":
    import argparse

    # for iter in [-1, 0, 100000, 200000]:
    for iter in [1000000]:

    # for iter in [-1]:
        # for seed in [0, 1,2,3,4]:
        # for seed in [5,6,7,8,9]:
        for seed in range(10):


            parser = argparse.ArgumentParser()
            parser.add_argument('--seed', '-s', type=int, default=seed)
            parser.add_argument('--iter', '-i', type=int, default=iter)

            args = parser.parse_args()

            # eval_ckpt(args)
            print(f'iter: {iter}', f'seed: {seed}')
            generate(args)
