from dizoo.box2d.lunarlander.config.lunarlander_cont_dqn_vqvae_ensemble_generation_config import main_config, create_config

from ding.entry import collect_episodic_demo_data, eval
import torch
import copy


def eval_ckpt(args):
    config = copy.deepcopy([main_config, create_config])
    eval(config, seed=args.seed, load_path=main_config.policy.model_path)
    # eval(config, seed=args.seed, load_path=main_config.policy.model_path, replay_path='/Users/puyuan/code/DI-engine/data_lunarlander_visualize/')


def generate(args):
    config = copy.deepcopy([main_config, create_config])
    state_dict = torch.load(main_config.policy.model_path, map_location='cpu')
    collect_episodic_demo_data(
        config,
        collect_count=1,
        seed=args.seed,
        expert_data_path='/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_crlw1_seed1_3M/data_iteration_best_1eps.pkl',
        # expert_data_path='/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_obs0_noema_smallnet_k8_upc50_seed1_3M/data_iteration_best_1eps.pkl',
        # expert_data_path='/Users/puyuan/code/DI-engine/data_lunarlander_visualize/dqn_sbh_ensemble20_noobs_noema_smallnet_k8_upc50_seed1_3M/data_iteration_best_1eps.pkl',
        state_dict=state_dict
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=1)
    args = parser.parse_args()

    eval_ckpt(args)
    # generate(args)

