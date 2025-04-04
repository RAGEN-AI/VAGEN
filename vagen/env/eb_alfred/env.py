import os
import time
import gym
import json
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List


# Import custom modules
from vagen.env.register import register
import vagen.env.eb_alfred.utils as utils
from vagen.env.eb_alfred.utils import alfred_objs, alfred_open_obj, alfred_pick_obj, alfred_slice_obj, alfred_open_obj, alfred_toggle_obj, alfred_recep
from vagen.env.eb_alfred.thor_connector import ThorConnector
from vagen.env.eb_alfred.data.preprocess import Dataset
from vagen.env.eb_alfred.gen import constants
from vagen.env.eb_alfred.main import logger
from vagen.env.base import BaseEnv, BaseInterface, IMAGE_PLACEHOLDER
from vagen.env.utils import convert_numpy_to_PIL, preprocess, postprocess

# global information
X_DISPLAY = '1'
#@TODO path
ALFRED_SPLIT_PATH = os.path.join(os.getcwd(), 'data/splits/splits.json')
ALFRED_REWARD_PATH = os.path.join(os.getcwd(), 'models/config/rewards.json')
ALFRED_DATASET_PATH = os.path.join(os.getcwd(), 'data/json_2.1.0')
ValidEvalSets = [
    'base', 'common_sense', 'complex_instruction', 'spatial', 
    'visual_appearance', 'long_horizon'
]


def get_global_action_space():
    """
    Generate a comprehensive action space for the environment.
    
    Returns:
        list: A list of supported action strings for various object interactions
    """
    action_space = []
    
    # Generate find actions for all objects
    findable_objs = alfred_objs
    action_space.extend([f"find a {obj}" for obj in findable_objs])
    
    # Generate pickup, putdown, and drop actions
    pickup_objs = alfred_pick_obj
    for obj in pickup_objs:
        action_space.extend([
            f"pick up the {obj}", 
        ])
    
    action_space.extend([
            f"put down the object in hand", 
            f"drop the object in hand"
        ])
    
    # Generate open/close actions
    open_objs = alfred_open_obj
    for obj in open_objs:
        action_space.extend([
            f"open the {obj}", 
            f"close the {obj}"
        ])
    
    # Generate toggle actions
    turn_on_objs = alfred_toggle_obj
    for obj in turn_on_objs:
        action_space.extend([
            f"turn on the {obj}", 
            f"turn off the {obj}"
        ])
    
    # Generate slice actions
    slice_objs = alfred_slice_obj
    action_space.extend([f"slice the {obj}" for obj in slice_objs])
    
    return action_space


class ALFREDEnv(BaseEnv, gym.Env):
    """
    Custom environment for ALFRED tasks using the BaseEnv interface.
    
    Attributes:
        env (ThorConnector): Interface for AI2THOR interactions
        action_space (gym.spaces.Discrete): Discrete action space 
        language_skill_set (list): Readable action descriptions
    """
    def __init__(self, **kwargs):
        """
        Initialize the ALFRED environment.
        """
        BaseEnv.__init__(self)
        
        # Extract configuration parameters
        self.eval_set = kwargs.pop('eval_set', 'base')
        self.exp_name = kwargs.pop('exp_name', '')
        self.down_sample_ratio = kwargs.pop('down_sample_ratio', 1.0)
        self.selected_indexes = kwargs.pop('selected_indexes', [])
        self.detection = kwargs.pop('detection_box', False)
        self.resolution = kwargs.pop('resolution', 500)
        
        # Setup paths and environment
        self.data_path = ALFRED_SPLIT_PATH
        self.reward_config_path = ALFRED_REWARD_PATH
        self.env = ThorConnector(
            x_display=X_DISPLAY, 
            player_screen_height=self.resolution, 
            player_screen_width=self.resolution
        )

        # Load dataset
        assert self.eval_set in ValidEvalSets
        self.dataset = self._load_dataset()
        if len(self.selected_indexes):
            self.dataset = [self.dataset[i] for i in self.selected_indexes]
        
        # Episode tracking
        self.number_of_episodes = len(self.dataset)
        self._reset_flag = False
        self._current_episode_num = 0
        self._initial_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 30
        self._cur_invalid_actions = 0
        self._max_invalid_actions = 10
        self._episode_start_time = 0
        self.episode_log = []
        
        # Task-related attributes
        self.episode_language_instruction = ''
        self.episode_data = None
        
        # Initialize action space
        self.language_skill_set = get_global_action_space()
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))
        
        # Environment feedback and logging
        self.feedback_verbosity = 0  # 0: concise, 1: verbose
        self.log_path = f'running/eb_alfred/{self.exp_name}'
        
        # Object mappings
        self.name_to_id_dict = None
        self.id_to_name_dict = None
        
        # Current index tracking for sequential tasks
        self.current_task_idx = 0

    def _load_dataset(self):
        """
        Load the dataset for the specified evaluation set.
        
        Returns:
            list: List of task dictionaries
        """
        with open(self.data_path) as f:
            dataset_split = json.load(f)
        dataset = dataset_split[self.eval_set]
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        return dataset

    def generate_additional_action_space(self):
        """
        Generate additional actions for receptacles with multiple instances.
        Updates the action space with scene-specific actions.
        """
        # Generate pickup, putdown, and drop actions
        add_findable_objs = []
        add_pickable_objs = []

        recept_obj_dict = {}
        pickable_obj_dict = {}
        name_to_id_dict = {}
        
        # Collect object information
        for obj in self.env.last_event.metadata['objects']:
            if obj['receptacle']:
                if obj['objectType'] in recept_obj_dict:
                    recept_obj_dict[obj['objectType']].append(obj['objectId']) 
                else:
                    recept_obj_dict[obj['objectType']] = [obj['objectId']]
            elif obj['pickupable']:
                if obj['objectType'] in pickable_obj_dict:
                    pickable_obj_dict[obj['objectType']].append(obj['objectId'])
                else:
                    pickable_obj_dict[obj['objectType']] = [obj['objectId']]

        # Store mapping for objects with multiple instances
        for key in recept_obj_dict:
            if len(recept_obj_dict[key]) >= 2:
                for i in range(len(recept_obj_dict[key])):
                    if i == 0:
                        name_to_id_dict[key] = recept_obj_dict[key][i]
                    else:
                        name_to_id_dict[key + '_{}'.format(i+1)] = recept_obj_dict[key][i]
                        add_findable_objs.append(key + '_{}'.format(i+1))
        
        for key in pickable_obj_dict:
            if len(pickable_obj_dict[key]) >= 2:
                for i in range(len(pickable_obj_dict[key])):
                    if i == 0:
                        name_to_id_dict[key] = pickable_obj_dict[key][i]
                    else:
                        name_to_id_dict[key + '_{}'.format(i+1)] = pickable_obj_dict[key][i]
                        add_pickable_objs.append(key + '_{}'.format(i+1))

        # Create inverse mapping
        id_to_name_dict = {}
        for key in name_to_id_dict:
            id_to_name_dict[name_to_id_dict[key]] = key

        # Generate additional actions
        add_findable_objs = sorted(list(set(add_findable_objs)))
        add_pickable_objs = sorted(list(set(add_pickable_objs)))
        action_space = [f"find a {obj}" for obj in add_findable_objs]
        
        for obj in add_findable_objs:
            if obj.split('_')[0] in alfred_open_obj:
                action_space.extend([
                    f"open the {obj}", 
                    f"close the {obj}"
                ])
                
        for obj in add_pickable_objs:
            if obj.split('_')[0] in alfred_pick_obj:
                action_space.extend([
                    f"find a {obj}", 
                ])

        # Update action space
        self.language_skill_set = get_global_action_space() + action_space
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))
        self.name_to_id_dict = name_to_id_dict
        self.id_to_name_dict = id_to_name_dict

    def _reset_controller(self, task):
        """
        Restore scene from a task specification and prepare the environment.
        
        Args:
            task (dict): Task specification including scene and initial state
        """
        traj_data = utils.load_task_json(task)
        traj_data['turk_annotations']['anns'][task['repeat_idx']]['task_desc'] = task["instruction"] 
        self.episode_data = traj_data
        
        # Set up arguments for the environment
        args_dict = {
            'data': ALFRED_DATASET_PATH, 
            'pframe': 300, 
            'fast_epoch': False,
            'use_templated_goals': False, 
            'dout': 'exp/model', 
            'pp_folder': 'pp',
            'reward_config': self.reward_config_path, 
            'max_steps': 1000
        }
        model_args = utils.dotdict(args_dict)
        
        # Extract scene configuration
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        # Restore scene
        scene_name = f'FloorPlan{scene_num}'
        self.episode_language_instruction = task["instruction"] 
        logger.info(f"Restoring scene {scene_name}...")
        self.env.reset(scene_name)
        self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        
        # Apply initial action
        if traj_data['scene']['init_action']['action'] == 'TeleportFull':
            del traj_data['scene']['init_action']["rotateOnTeleport"]
            traj_data['scene']['init_action']["standing"] = True
        self.env.step(dict(traj_data['scene']['init_action']))
        
        # Set task
        self.env.set_task(
            traj_data, 
            model_args, 
            reward_type='dense', 
            max_episode_length=self._max_episode_steps
        )
        
        # Generate additional actions for this specific scene
        self.generate_additional_action_space()

    def _reset(self, seed=None):
        """
        Reset the environment for a new episode.
        
        Args:
            seed (int, optional): Random seed for deterministic behavior
            
        Returns:
            dict: Initial observation
        """
        # If seed is provided, use it to determine the episode index
        if seed is not None:
            self._current_episode_num = seed % self.number_of_episodes
        
        # Reset the controller with the current episode
        self._reset_controller(self.dataset[self._current_episode_num])
        self._current_step = 0
        self._cur_invalid_actions = 0
        self._current_episode_num = (self._current_episode_num + 1) % self.number_of_episodes
        
        # Reset episode tracking
        self._reset_flag = True
        self.episode_log = []
        self._episode_start_time = time.time()
        
        # Return initial observation
        return {
            'head_rgb': self.env.last_event.frame,
        }
    
    def _render(self, mode='rgb_array'):
        """
        Render the current state of the environment.
        
        Args:
            mode (str): Rendering mode ('rgb_array' for image)
            
        Returns:
            numpy.ndarray: RGB image of the current scene
        """
        if mode == 'rgb_array':
            return self.env.last_event.frame
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def _step(self, action):
        """
        Execute a single environment step.
        
        Args:
            action: The action to take (int index or string action)
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        assert self._reset_flag, 'Reset environment before stepping'
        info = {}
        self._current_step += 1
        
        # Convert action to language action
        if isinstance(action, int):
            lang_action = self.language_skill_set[action]
        elif isinstance(action, str):
            lang_action = action
        else:
            raise NotImplementedError("Action must be an integer index or string action")

        # Process actions with object references
        if 'find' in lang_action or 'open' in lang_action or 'close' in lang_action:
            lang_action_split = lang_action.split(' ')
            if (self.name_to_id_dict is not None) and lang_action_split[-1] in self.name_to_id_dict:
                # Handle multiple instances by using the object ID
                lang_action = ' '.join(lang_action_split[:-1] + [self.name_to_id_dict[lang_action_split[-1]]])

        # Execute action
        event = self.env.llm_skill_interact(lang_action)
        if not event['success']:
            self._cur_invalid_actions += 1
        
        # Calculate reward and check task completion
        reward, done = self.env.get_transition_reward()
        subgoal_met = self.env.get_goal_conditions_met()
        
        # Prepare observation and info
        obs = {
            'head_rgb': self.env.last_event.frame,
        }
        
        # Check termination conditions
        if (self._current_step >= self._max_episode_steps or 
            self.env.get_goal_satisfied() or 
            self._cur_invalid_actions >= self._max_invalid_actions):
            done = True
        
        # Populate info dictionary
        info['task_success'] = float(self.env.get_goal_satisfied())
        info['task_progress'] = subgoal_met[0] / subgoal_met[1]
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['last_action_success'] = float(event['success'])
        info['object_states'] = {
            "cooled_objects": self.env.cooled_objects,
            "heated_objects": self.env.heated_objects,
            "cleaned_objects": self.env.cleaned_objects,
            "visible_objs": [obj['objectType'] for obj in self.env.last_event.metadata['objects'] if obj['visible']]
        }
        
        # Record action information
        if isinstance(action, int):
            info['action_id'] = action
            info['action_description'] = self.language_skill_set[action]
        else:
            info['action_id'] = -1  # Custom action
            info['action_description'] = action
        
        # Log episode information
        self.episode_log.append(info)
        
        return obs, reward, done, info
    
    def get_env_feedback(self, info):
        """
        Generate feedback message for the current step.
        
        Args:
            info (dict): Action execution information
            
        Returns:
            str: Descriptive message about step outcome
        """
        msg = ''
        if info["success"]:
            msg += "Last action executed successfully."
        else:
            if 'is not visible' in info['message'] and '|' in info['message']:
                # Format message for hidden objects
                recep_id = info['message'].split('because it is in ')[1].split('. Note')[0]
                if recep_id not in self.id_to_name_dict:
                    pos = recep_id.split('|')[0]
                else:
                    pos = self.id_to_name_dict[recep_id]
                message = info['message'].split(recep_id)[0] + pos + '. Go there to pick the object instead.'
            else:
                message = info['message']
            msg += f"Last action is invalid. {message}"
        return msg

    def close(self):
        """
        Clean up resources used by the environment.
        """
        self.env.stop()



@register(name="alfred")
class ALFREDInterface(BaseInterface):
    """Interface for ALFRED environment that implements BaseInterface."""
    
    def __init__(self, env_config: Dict, interface_config: Dict = None):
        """
        Initialize the ALFRED interface.
        
        Args:
            env_config: Configuration for the ALFRED environment
            interface_config: Configuration for the interface
        """
        super().__init__(env_config, interface_config)
        
        # Initialize environment
        self.env = ALFREDEnv(**self.env_config)
        
        # Extract interface configuration or set defaults
        self.format_reward = interface_config.get('format_reward', 0.5)
        self.max_action_per_step = interface_config.get('max_action_per_step', 1)
        self.max_action_penalty = interface_config.get('max_action_penalty', -0.1)
        
        # Initialize trajectory reward
        self.traj_reward = 0.0
        
        # Templates for prompts
        self.instruction_template = (
            "You are an agent in a household environment.\n"
            "Your task is: {task_instruction}\n\n"
            "You can take the following actions:\n"
            "- Find objects: 'find a [object]'\n"
            "- Pick up objects: 'pick up the [object]'\n"
            "- Put down objects: 'put down the object in hand'\n"
            "- Open/close receptacles: 'open the [receptacle]', 'close the [receptacle]'\n"
            "- Toggle devices: 'turn on the [device]', 'turn off the [device]'\n"
            "- Slice objects: 'slice the [object]'\n\n"
            "You can take up to {max_action_per_step} action(s) at a time.\n"
            "Provide your reasoning and actions in this format:\n"
            "<think>Your reasoning here...</think>\n"
            "<answer>action1, action2, ...</answer>"
        )
        
        self.observation_template = (
            "Current observation:\n{observation}\n\n"
            "Current task: {task_instruction}\n\n"
            "Decide your next action(s).\n"
            "Your response should be in the format of <think>...</think><answer>...</answer>"
        )
        
        self.action_template = (
            "After your answer, the executed action(s): {valid_action}.\n"
            "Current observation:\n{observation}\n\n"
            "Current task: {task_instruction}\n"
            "Reward: {reward}\n"
            "Done: {done}\n\n"
            "Decide your next action(s).\n"
            "Your response should be in the format of <think>...</think><answer>...</answer>"
        )

    def _extract_action(self, text: str) -> str:
        """
        Extract action from text.
        
        Args:
            text: Action text from LLM
            
        Returns:
            The extracted action or empty string if invalid
        """
        # Clean the input text
        text = text.strip().lower()
        
        # Check if action is valid by matching against language_skill_set
        for action in self.env.language_skill_set:
            if text == action.lower():
                return action
        
        return ""  # Invalid action

    def _reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Seed for the environment
            
        Returns:
            Initial observation and info
        """
        # Reset the environment
        obs = self.env.reset(seed=seed)
        
        # Reset trajectory reward
        self.traj_reward = 0.0
        
        # Get RGB image
        env_state = obs.get('head_rgb', None)
        if env_state is not None:
            env_state = convert_numpy_to_PIL(env_state)
        
        # Create observation dictionary
        text_template = self.observation_template.format(
            observation=IMAGE_PLACEHOLDER,
            task_instruction=self.env.episode_language_instruction
        )
        
        observation = {
            'text_template': text_template,
            'multi_modal_data': {
                IMAGE_PLACEHOLDER: [env_state],
            },
        }
        
        return observation, {}

    def _step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action string in the environment.
        
        Args:
            action: Action string from LLM
            
        Returns:
            Observation, reward, done, info
        """
        # Ensure environment isn't already finished
        if self.env._current_step >= self.env._max_episode_steps:
            return {
                'text_template': "Environment already finished.",
                'multi_modal_data': {}
            }, 0.0, True, {'llm_raw_response': action}
        
        # Preprocess the action text
        preprocess_result = preprocess(action, self._extract_action, "")
        
        # Calculate format reward
        reward = 0.0
        if preprocess_result.action_list:
            reward += self.format_reward
        
        # Apply max action penalty if needed
        if len(preprocess_result.action_list) > self.max_action_per_step:
            reward += self.max_action_penalty
            preprocess_result.action_list = preprocess_result.action_list[:self.max_action_per_step]
        
        # Execute actions
        final_info = {'llm_raw_response': preprocess_result.llm_raw_response}
        done = False
        
        for act in preprocess_result.action_list:
            if done:
                break
                
            # Execute action in environment
            obs, env_reward, done, info = self.env.step(act)
            reward += env_reward
            final_info.update(info)
        
        # Update trajectory reward
        self.traj_reward += reward
        
        # Get RGB image for observation
        env_state = obs.get('head_rgb', None)
        if env_state is not None:
            env_state = convert_numpy_to_PIL(env_state)
        
        # Create observation
        text_template = self.action_template.format(
            valid_action=", ".join(preprocess_result.action_list) if preprocess_result.action_list else "None",
            observation=IMAGE_PLACEHOLDER,
            task_instruction=self.env.episode_language_instruction,
            reward=f"{reward:.2f}",
            done=str(done)
        )
        
        observation = {
            'text_template': text_template,
            'multi_modal_data': {
                IMAGE_PLACEHOLDER: [env_state],
            },
        }
        
        return observation, reward, done, final_info

    def get_task_instruction(self) -> str:
        """
        Get the task instruction.
        
        Returns:
            Task instruction string
        """
        return self.instruction_template.format(
            task_instruction=self.env.episode_language_instruction,
            max_action_per_step=self.max_action_per_step
        )
    
    def get_traj_reward(self) -> float:
        """
        Get the total trajectory reward.
        
        Returns:
            Total trajectory reward
        """
        return self.traj_reward
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @classmethod
    def config_repr(cls, env_config: Dict, interface_config: Dict) -> str:
        """
        Create a string representation of the configuration.
        
        Args:
            env_config: Environment configuration dictionary
            interface_config: Interface configuration dictionary
            
        Returns:
            String representation of the configuration
        """
        env_str = f"ALFREDEnv(eval_set='{env_config.get('eval_set', 'base')}', " \
                 f"resolution={env_config.get('resolution', 500)}, " \
                 f"down_sample_ratio={env_config.get('down_sample_ratio', 1.0)})"
                 
        interface_str = f"ALFREDInterface(max_action_per_step={interface_config.get('max_action_per_step', 1)}, " \
                        f"format_reward={interface_config.get('format_reward', 0.5)})"
                        
        return f"{env_str}, {interface_str}"