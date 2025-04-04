import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets
import multiprocessing as mp
from functools import partial
import numpy as np
from vagen.env.create_dataset import DatasetCreator
from vagen.env.navigation.env import NavigationInterface

class NavigationDatasetCreator(DatasetCreator):
    def create_navigation_dataset(
        self, 
        seed: int = 0,
        train_ratio: float = 0.8,
        n_candidate: int = 100,
        force_gen: bool = False,
    ):
        train_file_path = os.path.join(self.data_dir, 'train.parquet')
        test_file_path = os.path.join(self.data_dir, 'test.parquet')
        
        # Check if files already exist and force_gen is False
        if not force_gen and os.path.exists(train_file_path) and os.path.exists(test_file_path):
            print(f"Dataset files already exist at {self.data_dir}. Skipping generation.")
            print(f"Use --force-gen to override and regenerate the dataset.")
            return
        seeds = range(seed, seed + n_candidate)
        train_size = int(len(seeds) * train_ratio)
        test_size = len(seeds) - train_size
        print(f"Train size: {train_size}, Test size: {test_size}")
        self.create_dataset(seeds, train_size, test_size, force_gen=force_gen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--n_candidate', type=int, default=100)
    parser.add_argument('--max_action_length', type=int, default=None)
    parser.add_argument('--force-gen', action='store_true', 
                        help='Force dataset generation even if files already exist')
    parser.add_argument('--data_dir', type=str, default='data/frozen_lake',)

    parser.add_argument('--resolution', type=int, default=500,
                    help='Resolution of the generated image')
    parser.add_argument('--eval_set', type=str, default='base',
                    help='Dataset that will be evaluated')
    parser.add_argument('--exp_name', type=str, default='test_base',
                    help='Experiment name')
    parser.add_argument('--down_sample_ratio', type=float, default=1.0,
                    help='Downsample ratio')
    parser.add_argument('--fov', type=int, default=100,
                    help='Field of view') 
    parser.add_argument('--multiview', action='store_true',
                    help='use multiview data')
    
    parser.add_argument('--visual_env', action='store_true',
                        help='Whether to use visual environment')
    
    parser.add_argument('--max_action_per_step', type=int, default=1,
                        help='Maximum number of actions per step')
    parser.add_argument('--max_action_penalty', type=float, default=-0.1,
                        help='Penalty for exceeding the maximum number of actions per step')
    parser.add_argument('--format_reward', type=float, default=0.5,
                        help='Reward for correct formatting')
    
    import os
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '0'
        print("Set PYTHONHASHSEED to 0 for reproducibility")
    else:
        print(f"PYTHONHASHSEED already set to {os.environ['PYTHONHASHSEED']}")
    

    args = parser.parse_args()
    args.name = 'navigation'
    args.env_config = {
        'resolution': args.resolution,
        'eval_set': args.eval_set,
        'exp_name': args.exp_name,
        'down_sample_ratio': args.down_sample_ratio,
        'fov': args.fov,
        'multiview': args.multiview,
        'visual_env': args.visual_env
    }
    args.interface_config = {
        'max_action_per_step': args.max_action_per_step,
        'max_action_penalty': args.max_action_penalty,
        'format_reward': args.format_reward,
    }
    creator = NavigationDatasetCreator(config=vars(args))
    creator.create_navigation_dataset(
        seed=args.start_seed,
        train_ratio=args.train_ratio,
        force_gen=args.force_gen, 
        n_candidate=args.n_candidate)
