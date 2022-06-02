from dizoo.mujoco.config.hopper_sac_data_generation_config_vqvae import main_config, create_config
# from dizoo.mujoco.config.hopper_td3_data_generation_config_vqvae import main_config, create_config
from ding.entry import collect_episodic_demo_data, eval, episode_to_transitions, episode_to_transitions_pure_expert
import torch
import copy


def eval_ckpt(args):
    config = copy.deepcopy([main_config, create_config])
    # eval(config, seed=args.seed, load_path=main_config.policy.learn.learner.load_path, replay_path='/home/puyuan/hopper_sac_seed0/')
    eval(config, seed=args.seed, load_path=main_config.policy.learn.learner.load_path)


def generate(args):
    config = copy.deepcopy([main_config, create_config])
    state_dict = torch.load(main_config.policy.learn.learner.load_path, map_location='cpu')
    collect_episodic_demo_data(
        config,
        collect_count=main_config.policy.other.replay_buffer.replay_buffer_size,
        seed=args.seed,
        expert_data_path=main_config.policy.collect.save_path,
        state_dict=state_dict
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    # eval_ckpt(args)
    generate(args)
    episode_to_transitions(data_path=main_config.policy.collect.save_path, expert_data_path='/home/puyuan/hopper_sac_seed0/expert_data_transitions_1000eps.pkl', nstep=3)
    episode_to_transitions_pure_expert(data_path=main_config.policy.collect.save_path, expert_data_path='/home/puyuan/hopper_sac_seed0/expert_data_transitions_1000eps_lt3500.pkl', nstep=3)

