from datasets import load_dataset, Dataset
from vagen.env.create_dataset import DatasetCreator
from typing import Union, List, Dict, Optional
import os

class SVGDatasetCreator(DatasetCreator):
    """Dataset creator for SVG data, adapting to Hugging Face dataset format"""

    def __init__(self, config: Dict):
        super().__init__(config)
        # Save dataset name/path
        self.dataset_name = self.env_config.get('dataset_name', 'default_dataset')
        self.seed = self.env_config.get('seed', '16')
        
    def create_dataset(
        self, 
        seed: Union[int, List[int]] = 0, 
        train_samples: Optional[int] = None,
        test_samples: Optional[int] = None, 
        force_gen: bool = False
    ):
        """
        Create SVG dataset utilizing Hugging Face dataset's built-in splits
        
        Args:
            force_gen: If True, regenerate the dataset even if files already exist
            train_samples: Number of training samples, use all if None
            test_samples: Number of test samples, use all if None
        """
        # Check output file paths
        train_file_path = os.path.join(self.data_dir, 'train.parquet')
        test_file_path = os.path.join(self.data_dir, 'test.parquet')
        
        # Check if files already exist
        if not force_gen and os.path.exists(train_file_path) and os.path.exists(test_file_path):
            print(f"Dataset files already exist at {self.data_dir}. Skipping generation.")
            print(f"Use --force-gen to override and regenerate the dataset.")
            return
            
        # Ensure directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load Hugging Face dataset
        try:
            print(f"Loading dataset {self.dataset_name}")
            hf_dataset = load_dataset(self.dataset_name)
            print(f"Dataset loaded successfully with splits: {list(hf_dataset.keys())}")
        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {e}")
            raise
        
        # Process each split
        splits = ["train", "test"]
        sample_limits = [train_samples, test_samples]
        output_files = [train_file_path, test_file_path]
        
        for split, sample_limit, output_file in zip(splits, sample_limits, output_files):
            if split in hf_dataset:
                # Get split data
                split_data = hf_dataset[split]
                print(f"Processing {split} split with {len(split_data)} examples")
                
                # If sample count specified, sample accordingly
                if sample_limit is not None:
                    sample_limit = min(sample_limit, len(split_data))
                    indices = list(range(sample_limit))
                    split_data = split_data.select(indices)
                    print(f"Sampled {sample_limit} examples from {split} split")
                
                # Create instances with environment configuration
                instances = []
                for idx, item in enumerate(split_data):
                    env_settings = {
                        'env_name': self.env_name,
                        'env_config': {
                            **self.env_config,
                            'svg_filename': item.get('Filename', f'image_{idx}'),
                            'svg_code': item.get('Svg', ''),
                            'item_idx': idx,
                            'data_dir': self.data_dir
                        },
                        'interface_config': self.interface_config,
                        'seed': self.seed
                    }
                    
                    instances.append({
                        "data_source": self.env_name,
                        "prompt": [{"role": "user", "content": ''}],
                        "extra_info": {"split": split, **env_settings}
                    })
                
                # Create dataset and save
                if instances:
                    try:
                        dataset = Dataset.from_list(instances)
                        dataset.to_parquet(output_file)
                        print(f"Created {split} dataset with {len(instances)} samples at {output_file}")
                    except Exception as e:
                        print(f"Error creating {split} dataset: {e}")
                        raise
                else:
                    print(f"No instances created for {split} split")
            else:
                print(f"Split '{split}' not found in dataset")
        
        print(f"Dataset successfully generated at {self.data_dir}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-gen', action='store_true', 
                      help='Force dataset generation even if files already exist')
    parser.add_argument('--data_dir', type=str, default='data/svg',
                      help='Directory to save the processed dataset')
    parser.add_argument('--dataset_name', type=str, default='starvector/svg-emoji-simple',
                      help='Hugging Face dataset name or path')
    parser.add_argument('--seed', type=int, default=16,
                        help='Seed for deterministic sampling')
    parser.add_argument('--model_size', type=str, default='large', choices=['small', 'base', 'large'],
                        help='Size of the DINO model to use')
    parser.add_argument('--dino_only', action='store_true',
                        help='Use only DINO score for reward')
    parser.add_argument("--analysis_mode", action='store_true',
                        help='Generate a seperate logs store all failure action')
    # @TODO customize sample number
    parser.add_argument('--train_samples', type=int, default=None,
                      help='Number of training samples to use')
    parser.add_argument('--test_samples', type=int, default=None, 
                      help='Number of test samples to use')
    
    
    # interface_config
    parser.add_argument('--max_action_per_step', type=int, default=1,
                        help='Maximum number of actions per step')
    parser.add_argument('--max_action_penalty', type=float, default=0,
                        help='Penalty for exceeding the maximum number of actions per step')
    parser.add_argument('--format_reward', type=float, default=0,
                        help='Reward for correct formatting')
    parser.add_argument('--format_penalty', type=float, default=0,
                        help='Penalty for incorrect formatting')

    # Optional score weights
    parser.add_argument('--dino_weight', type=float, default=None,
                        help='Weight for DINO score')
    parser.add_argument('--structural_weight', type=float, default=None,
                        help='Weight for structural accuracy')
    parser.add_argument('--color_weight', type=float, default=None,
                        help='Weight for color fidelity')
    parser.add_argument('--code_weight', type=float, default=None,
                        help='Weight for code efficiency')
    
    args, colab_par = parser.parse_known_args()

    args.name = 'svg'

    score_config = {
        'model_size': args.model_size,
        'dino_only': args.dino_only,
    }
    if args.dino_weight is not None: score_config['dino_weight'] = args.dino_weight
    if args.structural_weight is not None: score_config['structural_weight'] = args.structural_weight
    if args.color_weight is not None: score_config['color_weight'] = args.color_weight
    if args.code_weight is not None: score_config['code_weight'] = args.code_weight

    args.env_config = {
        'dataset_name': args.dataset_name,
        'data_dir': args.data_dir,
        'seed': args.seed,
        'score_config': score_config
    }
    
    args.interface_config = {
        'max_action_per_step': args.max_action_per_step,
        'max_action_penalty': args.max_action_penalty,
        'format_reward': args.format_reward,
        'format_penalty': args.format_penalty,
        'analysis_mode': args.analysis_mode
    }
    # Create dataset
    creator = SVGDatasetCreator(config=vars(args))
    creator.create_dataset(
        force_gen=args.force_gen,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )
