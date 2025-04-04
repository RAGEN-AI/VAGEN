from typing import Tuple
import ai2thor.controller
import gym
import numpy as np
import time
from PIL import Image
import json
import os
import sys
import math
import re
from ai2thor.platform import CloudRendering
from vagen.env.navigation.utils import draw_target_box, draw_boxes
import copy
from dataclasses import dataclass
from vagen.env.base import (
    BaseEnv,
    BaseInterface,
    IMAGE_PLACEHOLDER
)
from typing import Tuple, Dict, Optional, List, Any, Union
from ai2thor.platform import CloudRendering
from vagen.utils import NoLoggerWarnings
from vagen.utils import set_seed
from vagen.env.register import register
from vagen.env.utils import postprocess, convert_numpy_to_PIL, preprocess
from vagen.env.navigation.prompt import *

class NavigationEnv(BaseEnv, gym.Env):
    SUCCESS_THRESHOLD = 1

    ValidEvalSets = [
        'base', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon'
    ]


    DISCRETE_SKILLSET = [
        "Move forward by 0.25",
        "Move backward by 0.25",
        "Move rightward by 0.25",
        "Move leftward by 0.25",
        "Rotate to the right by 90 degrees.",
        "Rotate to the left by 90 degrees.",
        "Tilt the camera upward by 30 degrees.",
        "Tilt the camera downward by 30 degrees.",
        # "Crouch to be lower",
        # "Stand to be taller",
        # "Complete the current task."
    ]

    def __init__(self, **kwargs):
        BaseEnv.__init__(self)
        resolution = kwargs.pop('resolution', 500)
        eval_set=kwargs.pop('eval_set', 'base')
        exp_name=kwargs.pop('exp_name', 'test_base')
        down_sample_ratio=kwargs.pop('down_sample_ratio', 1.0)
        fov = kwargs.pop('fov', 100)
        multiview = kwargs.pop('multiview', False)

        self.resolution = resolution
        self.config = {
            "agentMode": "default",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "width": self.resolution,
            "height": self.resolution,
            "fieldOfView": fov,
            "platform": CloudRendering
        }
        self.env = ai2thor.controller.Controller(**self.config)
        # load dataset
        assert eval_set in self.ValidEvalSets
        self.down_sample_ratio = down_sample_ratio
        self.data_path = os.path.join(os.path.dirname(__file__), f"datasets/{eval_set}.json")
        self.dataset = self._load_dataset()
        
        # Episode tracking
        self.number_of_episodes = len(self.dataset)
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 20
        self._episode_start_time = 0
        self.is_holding = False
        self.episode_log = []
        self.episode_language_instruction = ""
        self.episode_data = None

        self._last_event = None

        self.standing = True

        # set action space
        self.language_skill_set = self.DISCRETE_SKILLSET
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))
        
        # set log and verbosity(0 for concise)
        self.feedback_verbosity = 0
        self.log_path = 'running/eb_nav/{}'.format(exp_name)

        self.multiview = multiview
        self.img_paths = []

    def _load_dataset(self):
        with open(self.data_path) as f:
            dataset_split = json.load(f)
        dataset = dataset_split["tasks"]
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        return dataset
    
    def _reset(self, seed: int):
        """
        Reset the environment.

        :param scene: Optionally set the scene for reset.
        :return: The initial observation.
        """
        # self.save_episode_log()
        # assert self._current_episode_num < self.number_of_episodes
        idx = seed % self.number_of_episodes
        # start reset environment 
        traj_data = self.dataset[idx]
        self.episode_data = traj_data
        self.episode_language_instruction = traj_data["instruction"]

        scene_name = traj_data["scene"]
        # logger.info(f"Restoring scene {scene_name}...")
        self._last_event = self.env.reset(
            scene=scene_name
        )

        if self.multiview:
            event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = copy.deepcopy(event.metadata["actionReturn"])
            pose["orthographic"] = True

            # add the camera to the scene
            self.env.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )

        pose = traj_data["agentPose"]
        self.env.step(
            action="Teleport",
            position={
                "x": pose["position"]["x"],
                "y": pose["position"]["y"],
                "z": pose["position"]["z"]
            },
            rotation={
                "x": 0,
                "y": pose["rotation"],
                "z": 0
            },
            horizon=pose["horizon"],
            standing=True
        )

        # finish reset environment 
        # reset episode information
        self._current_step = 0

        self.standing = True
        obs = {
            'head_rgb': self.env.last_event.frame
        }
        self.episode_log = []
        self._episode_start_time = time.time()

        self.img_paths = []

        return obs
    
    def _render(self, mode='rgb_array'):
        assert mode == 'rgb_array', "Navigation env can only support RGB image"
        return self.env.last_event.frame

    def _step(self, action: int, reasoning=None):
        """
        Perform an action in the environment.

        :param action: The name of the action to perform.
        :param kwargs: Additional parameters for the action.
        :return: Event.
        """
        prev_pos = self.env.last_event.metadata["agent"]["position"]
        # assert self._reset, 'Reset env before stepping'
        info = {}
        
        self._current_step += 1

        if self._current_step >= self._max_episode_steps:

            if type(action) != int or action > 8 or action < 1:
                action = np.random.randint(1, 9)

            self.discrete_action_mapper(action)
            reward, distance = self.measure_success()
            done = True
            info['action_description'] = self.language_skill_set[action-1]

        else:
            if type(action)!=int or action > 8 or action < 1:
                action = np.random.randint(1, 9)

            self.discrete_action_mapper(action)
            reward, distance = self.measure_success()
            if reward > 0:
                done = True
            else:
                done = False
            info['action_description'] = self.language_skill_set[action-1]

        curr_pos = self.env.last_event.metadata["agent"]["position"]

        info["action_is_effective"] = curr_pos["x"] != prev_pos["x"] or curr_pos["z"] != prev_pos["z"]

        info['action_description'] = self.language_skill_set[action]

        obs = {'head_rgb': self.env.last_event.frame}
        reward, distance = self.measure_success()

        ## test calculate reward
        info['distance'] = distance
        info['env_feedback'] = self.get_env_feedback(self._last_event)
        info['reasoning'] = reasoning
        # info['reflection'] = reasoning['reasoning_and_reflection']
        # info['plan'] = reasoning['language_plan']
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['task_success'] = reward
        info['last_action_success'] = self.env.last_event.metadata['lastActionSuccess']
        info['action_id'] = action 

        return obs, reward, done, info
    
    def get_env_feedback(self, event):
        """
        To extract relevant information from the event to construct a feedback dictionary.

        :param event: self._last_event
        :return: A dictionary containing structured feedback.
        """
        if self.feedback_verbosity == 1:
            feedback = {
                "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
                "errorMessage": event.metadata.get("errorMessage", None),
                "lastAction": event.metadata.get("lastAction", None),

                "agent": {
                    "position": event.metadata.get("agent", {}).get("position", {}),
                    "rotation": event.metadata.get("agent", {}).get("rotation", {}),
                    "is_standing": self.standing
                }
            }
        else:
            # Does not provide the specific reason why the action fails if so
            feedback = {
                "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
                "lastAction": event.metadata.get("lastAction", None),
                "errorMessage": event.metadata.get("errorMessage", None),

                "agent": {
                    "is_standing": self.standing
                }
            }

        msg = ''
        if feedback["lastActionSuccess"]:
            msg += f"Last action {feedback['lastAction']} executed successfully."
        else:
            msg += f"Last action {feedback['lastAction']} is invalid. {feedback['errorMessage']}"
        return msg
    
    def measure_success(self):
        # success measurement
        agent_position = self.env.last_event.metadata["agent"]["position"]
        target_object_id = self.episode_data["targetObjectIds"]
        target_position = self.episode_data["target_position"]
        dist = math.sqrt(
            (agent_position["x"] - target_position["x"])**2 +
            (agent_position["z"] - target_position["z"])**2
        )
        success = (dist <= self.SUCCESS_THRESHOLD)
        return float(success), dist
    
    def discrete_action_mapper(self, action_index):
        """
        Maps a discrete action index to the corresponding iTHOR environment action.

        Parameters:
            env: The AI2-THOR environment object.
            action_index: An integer representing the action index.

        Raises:
            ValueError: If the action index is invalid.
        """

        if action_index == 1:  # Move forward by 0.25 meter
            self._last_event = self.env.step(action="MoveAhead", moveMagnitude=0.25)
        elif action_index == 2:  # Move backward by 0.25 meter
            self._last_event = self.env.step(action="MoveBack", moveMagnitude=0.25)
        elif action_index == 3:  # Move right by 0.25 meter
            self._last_event = self.env.step(action="MoveRight", moveMagnitude=0.25)
        elif action_index == 4:  # Move left by 0.25 meter
            self._last_event = self.env.step(action="MoveLeft", moveMagnitude=0.25)
        elif action_index == 5:  # Rotate clockwise by 45 degrees
            self._last_event = self.env.step(action="RotateRight", degrees=90)
        elif action_index == 6:  # Rotate counterclockwise by 45 degrees
            self._last_event = self.env.step(action="RotateLeft", degrees=90)
        elif action_index == 7:  # Tilt the camera upward by 30 degrees
            self._last_event = self.env.step(action="LookUp", degrees=30)
        elif action_index == 8:  # Tilt the camera downward by 30 degrees
            self._last_event = self.env.step(action="LookDown", degrees=30)

    def close(self):
        """Close the environment."""
        self.env.stop()

@dataclass
class PreprocessResult:
    action_list: List[int]
    answer_list: List[str] # string of extracted answer (may be invalid action)
    valid_list: List[bool]
    think: str
    answer: str
    llm_raw_response: str

    def to_dict(self):
        return {
            'action_list': self.action_list,
            'answer_list': self.answer_list,
            'valid_list': self.valid_list,
            'think': self.think,
            'answer': self.answer,
            'llm_raw_response': self.llm_raw_response,
        }
    
@register(name="navigation")
class NavigationInterface(BaseInterface):
    INVALID_ACTION = 0
    ACTION_LOOKUP = {
        1: "moveahead",
        2: "moveback",
        3: "moveright",
        4: "moveleft",
        5: "rotateright",
        6: "rotateleft",
        7: "lookup",
        8: "lookdown",
    }
    def __init__(
            self,
            env_config: Dict,
            interface_config: Dict,
        ):
        super().__init__(env_config, interface_config)
        resolution = self.env_config['resolution']
        eval_set = self.env_config['eval_set']
        exp_name = self.env_config['exp_name']
        down_sample_ratio = self.env_config['down_sample_ratio']
        fov = self.env_config['fov']
        multiview = self.env_config['multiview']
        self.env = NavigationEnv(
            resolution=resolution,
            eval_set=eval_set,
            exp_name=exp_name,
            down_sample_ratio=down_sample_ratio,
            fov=fov,
            multiview=multiview,
        )
        self.visual_env = self.env_config.get('visual_env', True)

        max_action_per_step = interface_config.setdefault('max_action_per_step', 1)
        max_action_penalty = interface_config.setdefault('max_action_penalty', -0.5)
        format_reward = interface_config.setdefault('format_reward', 0.5)
        self.interface_config = {
            'max_action_per_step': max_action_per_step,
            'max_action_penalty': max_action_penalty,
            'format_reward': format_reward,
        }

    @classmethod
    def _extract_one_action(cls, text):
        """
        Extract action from text.
        NOTE: the action space is different from the original, with these mappings:
        - 0: Invalid Action
        - 1: MoveAhead
        - 2: MoveBack
        - 3: MoveRight
        - 4: MoveLeft
        - 5: RotateRight
        - 6: RotateLeft
        - 7: LookUp
        - 8: LookDown
        """
        ACTION_MAP = {
            "moveahead": 1,
            "moveback": 2,
            "moveright": 3,
            "moveleft": 4,
            "rotateright": 5,
            "rotateleft": 6,
            "lookup": 7,
            "lookdown": 8
        }
        
        # Updated pattern to match either the number (1-8) or action name (case insensitive)
        pattern = r'^\s*(([1-8])\s*\((moveahead|moveback|moveright|moveleft|rotateright|rotateleft|lookup|lookdown)\)|(moveahead|moveback|moveright|moveleft|rotateright|rotateleft|lookup|lookdown)|([1-8]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return cls.INVALID_ACTION 
        
        if match.group(2):   
            return int(match.group(2))
        elif match.group(4): 
            # Convert to lowercase to match the keys in ACTION_MAP
            action_name = match.group(4).lower()
            return ACTION_MAP[action_name]
        elif match.group(5): 
            return int(match.group(5))
        
        return cls.INVALID_ACTION
    
    def _step(self, raw_text: str) -> Tuple[Any, float, bool, Dict]:
        """Step the environment with llm raw response
        - Multiple actions are allowed, execute until the first invalid action or environment terminates
        - The observation is the last step observation
        
        Args:
            raw_text: raw text from LLM

        Returns:
            Tuple[Any, float, bool, Dict]: observation, reward, done, info
            - observation (dict): observation of the environment
            - reward (float): reward of the environment for the raw_text (multiple actions, including format reward and env reward)
            - done (bool): whether the environment is done
            - info (dict): extra info
        """
        flag, _ = self.env.measure_success()
        finished = int(flag)
        assert not finished, "Environment finished before step"
        reward, done, final_info = 0, False, {}

        # preprocess_result = self._preprocess(raw_text)
        preprocess_result = preprocess(raw_text, self._extract_one_action, self.INVALID_ACTION)
        think = preprocess_result.think
        action_list = preprocess_result.action_list
        answer = preprocess_result.answer
        final_info['llm_raw_response'] = preprocess_result.llm_raw_response

        # parse format and action list
        if action_list:
            reward += self.interface_config['format_reward']
        if len(action_list) > self.interface_config['max_action_per_step']:
            reward += self.interface_config['max_action_penalty']
            action_list = action_list[:self.interface_config['max_action_per_step']]
            preprocess_result.action_list = action_list

        info = {}
        for action in action_list:
            success, _ = self.env.measure_success()
            if done or int(success):
                break
            _, env_reward, done, info = self.env.step(action)
            # if env_reward == -0.1:
            #     env_reward = -0.01 # NOTE hard coded here to set step reward to 0
            reward += env_reward
        self.traj_reward += reward
        final_info.update(info) # NOTE currently only use the last step info

        env_state = self.env._render(mode='rgb_array') # NOTE currently called after step


        return postprocess(
            env_state=env_state,
            reward=reward,
            done=done,
            info=final_info,
            preprocess_result=preprocess_result,
            action_lookup=self.ACTION_LOOKUP,
            action_template=action_template,
        )
    
    def _reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and return the observation at the first step.
        """
        self.env._reset(seed=seed)
        self.traj_reward = 0
        env_state = self.env._render(mode='rgb_array') # NOTE currently called after reset
        if isinstance(env_state, np.ndarray):
            env_state = convert_numpy_to_PIL(env_state)
        observation = IMAGE_PLACEHOLDER if not isinstance(env_state, str) else env_state
        text_template = init_observation_template.format(
            observation=observation,
        )
        if isinstance(env_state, str):
            obs = {'text_template': text_template}
        else:
            obs = {
                'text_template': text_template,
                'multi_modal_data': {
                    IMAGE_PLACEHOLDER: [env_state],
                },
            }
        return obs, {}
    
    def close(self):
        self.env.close()

    @classmethod
    def config_repr(cls, config: Dict, interface_config: Dict) -> str:
        """
        Create a string representation of the configuration.
        
        Args:
            config: Dictionary containing configuration
            
        Returns:
            String representation of the configuration
            
        Raises:
            ValueError: If required keys are missing from the configuration

        resolution = self.env_config['resolution']
        eval_set = self.env_config['eval_set']
        exp_name = self.env_config['exp_name']
        down_sample_ratio = self.env_config['down_sample_ratio']
        fov = self.env_config['fov']
        multiview = self.env_config['multiview']
        """
        required_keys = ['resolution', 'eval_set', 'exp_name', 'down_sample_ratio', \
                         'fov', 'multiview']
        
        # Check for required keys
        if not all(key in config for key in required_keys):
            missing_keys = [key for key in required_keys if key not in config]
            raise ValueError(f"Missing required keys in config: {missing_keys}")
        
        env_config_str = (
            f"NavigationEnv(resolution={config['resolution']}, "
            f"eval_set={config['eval_set']}, "
            f"exp_name={config['exp_name']}, "
            f"down_sample_ratio={config['down_sample_ratio']}, "
            f"fov={config['fov']}, "
            f"multiview={config['multiview']}"
        )
        interface_config_str = (
            f"NavigationInterface(max_action_per_step={interface_config.get('max_action_per_step', 1)}, "
            f"format_reward={interface_config.get('format_reward', 0.5)})"
        )
        # Format the configuration string
        return f"{env_config_str}, {interface_config_str}"
    
    def get_task_instruction(self) -> str:
        return instruction_template.format(
            max_action_per_step=self.interface_config['max_action_per_step']
        )
    
    def get_traj_reward(self):
        return self.traj_reward