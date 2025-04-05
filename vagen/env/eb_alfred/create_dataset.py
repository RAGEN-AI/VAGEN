import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from vagen.env.create_dataset import DatasetCreator

class ALFREDDatasetCreator(DatasetCreator):
    def create_alfred_dataset(
        self,
        seed: int = 0,
        train_ratio: float = 0.8,
        n_candidate: int = 100,
        force_gen: bool = False,
    ):
        """
        Create dataset for ALFRED environment.
        
        Args:
            seed: Starting seed value
            train_ratio: Ratio of data to use for training
            n_candidate: Total number of examples to generate
            force_gen: Whether to force regeneration if files exist
        """
        train_file_path = os.path.join(self.data_dir, 'train.parquet')
        test_file_path = os.path.join(self.data_dir, 'test.parquet')
        
        # Check if files already exist and force_gen is False
        if not force_gen and os.path.exists(train_file_path) and os.path.exists(test_file_path):
            print(f"Dataset files already exist at {self.data_dir}. Skipping generation.")
            print(f"Use --force-gen to override and regenerate the dataset.")
            return
        
        # Generate the seed range for dataset
        seeds = range(seed, seed + n_candidate)
        train_size = int(len(seeds) * train_ratio)
        test_size = len(seeds) - train_size
        print(f"Train size: {train_size}, Test size: {test_size}")
        
        # Create the dataset with the specified parameters
        self.create_dataset(seeds, train_size, test_size, force_gen=force_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--n_candidate', type=int, default=100)
    parser.add_argument('--force-gen', action='store_true', 
                        help='Force dataset generation even if files already exist')
    parser.add_argument('--data_dir', type=str, default='data/eb_alfred')

    # ALFRED environment specific arguments
    parser.add_argument('--resolution', type=int, default=500,
                    help='Resolution of the generated image')
    parser.add_argument('--eval_set', type=str, default='base',
                    help='Dataset that will be evaluated')
    parser.add_argument('--exp_name', type=str, default='test_base',
                    help='Experiment name')
    parser.add_argument('--down_sample_ratio', type=float, default=1.0,
                    help='Downsample ratio')
    parser.add_argument('--detection_box', action='store_true',
                    help='Whether to show detection boxes in images')
    
    parser.add_argument('--max_action_per_step', type=int, default=1)
    
    parser.add_argument('--max_action_penalty', type=float, default=-0.1)
    
    parser.add_argument('--format_reward', type=float, default=0.5) 
    # Set fixed seed for reproducibility
    import os
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '0'
        print("Set PYTHONHASHSEED to 0 for reproducibility")
    else:
        print(f"PYTHONHASHSEED already set to {os.environ['PYTHONHASHSEED']}")
    
    # Parse arguments
    args = parser.parse_args()
    args.name = 'eb_alfred'
    
    # Set environment configuration
    args.env_config = {
        'resolution': args.resolution,
        'eval_set': args.eval_set,
        'exp_name': args.exp_name,
        'down_sample_ratio': args.down_sample_ratio,
        'detection_box': args.detection_box
    }
    
    # Interface configuration can be added if needed
    args.interface_config = {
        'max_action_per_step': args.max_action_per_step,
        'max_action_penalty': args.max_action_penalty,
        'format_reward': args.format_reward,
    }
    
    # Create and run the dataset creator
    creator = ALFREDDatasetCreator(config=vars(args))
    creator.create_alfred_dataset(
        seed=args.start_seed,
        train_ratio=args.train_ratio,
        force_gen=args.force_gen, 
        n_candidate=args.n_candidate)