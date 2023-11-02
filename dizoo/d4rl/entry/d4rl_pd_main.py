import os
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/mnt/afs/niuyazhe/code/.mujoco/mujoco210/bin:/usr/local/nvidia/lib64"
os.environ["MUJOCO_PY_MUJOCO_PATH"] = "/mnt/afs/niuyazhe/code/.mujoco/mujoco210"
os.environ["D4RL_DATASET_DIR"] = "/mnt/afs/niuyazhe/code/dataset/d4rl"
from ding.entry import serial_pipeline_offline
from ding.config import read_config
from pathlib import Path


def train(args):
    # launch from anywhere
    config = Path(__file__).absolute().parent.parent / 'config' / args.config 
    config = read_config(str(config))
    config[0].exp_name = config[0].exp_name.replace('0', str(args.seed))
    serial_pipeline_offline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--config', '-c', type=str, default='hopper_medium_expert_pd_config.py')
    args = parser.parse_args()
    train(args)