import gym
import numpy as np
import re
import copy
from typing import Tuple, Dict, Optional, List, Any, Union
from PIL import Image
from dataclasses import dataclass
from gymnasium.utils import seeding
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv

from vagen.utils import NoLoggerWarnings
from vagen.utils import set_seed
from vagen.env.register import register
from vagen.env.base import (
    BaseEnv,
    BaseInterface,
    IMAGE_PLACEHOLDER
)
from vagen.env.utils import postprocess, convert_numpy_to_PIL, preprocess
from vagen.env.frozen_lake.prompt import *

class FrozenLakeEnv(BaseEnv, GymFrozenLakeEnv):
     # Map gym state in integer
    MAP_LOOKUP = {
        b"P": 0,  # player
        b"F": 1,  # frozen
        b"H": 2,  # hole
        b"G": 3,  # goal
    }

    # Define rules to transform to rendered text observation of the environment
    GRID_LOOKUP = {
        0: " P \t",  # player
        1: " _ \t",  # frozen
        2: " O \t",  # hole
        3: " G \t",  # goal
        4: " X \t",  # player fall into hole
        5: " √ \t",  # player on goal
    }

    ACTION_LOOKUP = {
        0: "None",
        1: "Left",
        2: "Down",
        4: "Up",
        3: "Right",
    }

    def __init__(self, **kwargs):
        BaseEnv.__init__(self)
        desc = kwargs.pop('desc', None)                 # env map, might be a 2D array
        is_slippery = kwargs.pop('is_slippery', True)
        size = kwargs.pop('size', 8)
        p = kwargs.pop('p', 0.8)                        # probability of frozen tile
        seed = kwargs.pop('seed', None)
        self.desc = desc
        self.is_slippery = is_slippery
        self.size = size
        self.p = p
        self.seed = seed

        if desc is None:
            random_map = generate_random_map(size=size, p=p, seed=seed)
        else:
            random_map = np.asarray(copy.deepcopy(desc), dtype="c")
        GymFrozenLakeEnv.__init__(
            self,
            desc=random_map,
            is_slippery=is_slippery
        )
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1) 
        self.map_kwargs = {
            "size": size,
            "p": p,
        }
        self.env_kwargs = {
            "is_slippery": is_slippery,
            "desc": copy.deepcopy(desc),
            "seed": seed,
        }
        self.action_map = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        } # map from custom Env action to action defined in FrozenLakeEnv in gymnasium

        self.reward = 0
        self._valid_actions = []

    def _get_player_position(self):
        return (self.s // self.ncol, self.s % self.ncol) # (row, col)
    
    def _reset(
            self,
            seed: int,
            reset_map = True,
    ):
        """
        Reset the environment, there are two options:
        1. reset the map, generate a new map (reset_map=True)
        2. reset the environment with the same map, while putting the agent back to the start position (reset_map=False)
        Both can reset the seed
        NOTE if seed is the same, the map will be the same
        """
        if reset_map:
            self.__init__(
                size=self.map_kwargs["size"],
                p=self.map_kwargs["p"],
                seed=seed,
                is_slippery=self.env_kwargs["is_slippery"],
            )
        GymFrozenLakeEnv.reset(self, seed=seed)
        return self._render(mode='text') 
    
    def _render(self, mode='text'):
        assert mode in ['text', 'list', 'state', 'rgb_array', 'ansi']
        if mode in ['rgb_array', 'ansi']:
            prev_render_mode = self.render_mode
            self.render_mode = mode
            obs = GymFrozenLakeEnv.render(self)
            self.render_mode = prev_render_mode
            return obs
        room_state = copy.deepcopy(self.desc)

        # replace the position of start 'S' with 'F'
        position_S = np.where(room_state == b'S')
        room_state[position_S] = b'F'

        # replace the position of the player with 'P'
        position_P = self._get_player_position()
        room_state[position_P] = b'P'

        if mode == 'state':
            # transform 'S', 'F', 'H', 'G' to numpy integer array
            room_state = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room_state)
            # add player in hole or player on goal
            if self.desc[position_P] == b'H':
                room_state[position_P] = 4
            elif self.desc[position_P] == b'G':
                room_state[position_P] = 5
            return room_state
        
        room_state = self._render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'text':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
        
    def _step(self, action: int):
        """
        - Map custom action to gymnasium FrozenLakeEnv action and take the step
        - Check if the action is effective (whether player moves in the env).
        """
        assert isinstance(action, int), "Action must be an integer"
        assert not self.success(), "Agent has already reached the goal or hole"
        
        prev_player_position = int(self.s)
        with NoLoggerWarnings():
            player_pos, reward, done, _, prob = GymFrozenLakeEnv.step(self, self.action_map[action])

        # obs = self._render()
        return player_pos, reward, done, {"action_is_effective": prev_player_position != int(player_pos)}
    
    def success(self):
        """
        Check if the agent has reached the goal (G) or hole (H)
        """
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"G"

    def finished(self):
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"GH"
    
    def close(self):
        GymFrozenLakeEnv.close(self)


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

@register(name="frozen_lake")
class FrozenLakeInterface(BaseInterface):

    INVALID_ACTION = 0
    FORMAT_REWARD = 1
    ACTION_LOOKUP = {
        0: "None",
        1: "Left",
        2: "Down",
        3: "Right",
        4: "Up",
    }

    def __init__(
            self,
            env_config: Dict,
            interface_config: Dict,
        ):
        super().__init__(env_config, interface_config)

        desc = self.env_config['desc']
        is_slippery = self.env_config['is_slippery']
        size = self.env_config['size']
        p = self.env_config['p']
        seed = self.env_config.get('seed', 42)
        self.env = FrozenLakeEnv(
            desc=desc,
            is_slippery=is_slippery,
            size=size,
            p=p,
            seed=seed,
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
        NOTE: the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
        - 0: Still (Invalid Action)
        - 1: Left
        - 2: Down
        - 3: Right
        - 4: Up
        """
        DIRECTION_MAP = {"Left": 1, "Down": 2, "Right": 3, "Up": 4}
        # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
        # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
        pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return cls.INVALID_ACTION 
        
        if match.group(2):   
            return int(match.group(2))
        elif match.group(4): 
            return DIRECTION_MAP[match.group(4).capitalize()]
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

        assert not self.env.finished(), "Environment finished before step"
        reward, done, final_info = 0, False, {}


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
            if done or self.env.finished():
                break
            _, env_reward, done, info = self.env.step(action)
            # if env_reward == -0.1:
            #     env_reward = -0.01 # NOTE hard coded here to set step reward to 0
            reward += env_reward
        self.traj_reward += reward
        final_info.update(info) # NOTE currently only use the last step info

        env_state = self.env._render(mode='text' if not self.visual_env else 'rgb_array') # NOTE currently called after step

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
        env_state = self.env._render(mode='text' if not self.visual_env else 'rgb_array') # NOTE currently called after reset
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
        """
        required_keys = ['desc', 'is_slippery', 'size', 'p']
        
        # Check for required keys
        if not all(key in config for key in required_keys):
            missing_keys = [key for key in required_keys if key not in config]
            raise ValueError(f"Missing required keys in config: {missing_keys}")
            
        # Format the configuration string
        env_config_str = (
            f"FrozenLakeGame(desc={config['desc']}, "
            f"is_slippery={config['is_slippery']}, "
            f"size={config['size']}, "
            f"p={config['p']}"
        )
        interface_config_str = (
            f"SokobanInterface(max_action_per_step={interface_config.get('max_action_per_step', 1)}, "
            f"max_action_penalty={interface_config.get('max_action_penalty', -0.1)}, "
            f"format_reward={interface_config.get('format_reward', 0.5)})"
        )
        return f"{env_config_str}, {interface_config_str}"
    
    def get_task_instruction(self) -> str:
        return instruction_template
    
    def get_traj_reward(self):
        return self.traj_reward

def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    # generate random start and end points

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])

        while True:
            start_r = np_random.integers(0, size)
            start_c = np_random.integers(0, size)
            goal_r = np_random.integers(0, size)
            goal_c = np_random.integers(0, size)
            
            # Ensure start and goal are different positions
            if (start_r, start_c) != (goal_r, goal_c):
                break
            
        board[start_r][start_c] = "S"
        board[goal_r][goal_c] = "G"
        
        
        valid = is_valid(board, size)
    return ["".join(x) for x in board]

def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    # find the start point
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0]))
    # dfs to check if there is a path from start to goal
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False