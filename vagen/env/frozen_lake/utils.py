import random
import numpy as np
import marshal
import copy
from collections import deque
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from vagen.env.frozen_lake.env import FrozenLakeEnv

def get_shortest_action_path(
        seed,
        size=6,
        p=0.8,
        is_slippery=True,
        MAX_DEPTH=100
    ):
        """
        Get the shortest action path to push all boxes to the target spots.
        Use BFS to find the shortest path.
        NOTE use action sequence to recover the environment
        =========================================================
        Parameters:
            env: frozen lake environment
            MAX_DEPTH (int): the maximum depth of the search
        =========================================================
        Returns:
            action_sequence (list): the action sequence to push all boxes to the target spots
        """
        env = FrozenLakeEnv(size=size, p=p, is_slippery=is_slippery, seed=seed)
        env._reset(seed=seed)
        action_sequence, state = [], env.s
        queue = deque([(action_sequence, state)])
        explored_states = set()

        actions = [1, 2, 3, 4] 
        
        while queue:
            action_sequence, state = queue.popleft()
            if len(action_sequence) > MAX_DEPTH:
                return [] # No solution found

            if action_sequence:
            # reduce the search space by checking if the state has been explored
                state_tohash = marshal.dumps(state)
                if state_tohash in explored_states:
                    continue
                explored_states.add(state_tohash)
            
                
            # Try each direction
            for a in actions:
                # recover the environment
                env._reset(seed=seed)
                for prev_a in action_sequence:
                    env.step(prev_a)


                obs, reward, done, info = env._step(a)
                # print(f"action: {a}, reward: {reward}, done: {done}")
                # print(f"obs: {obs}")
                if done and reward > 0: # succeed
                    return action_sequence + [a]
                elif done and reward == 0: # failed
                    continue
                else:
                    queue.append((action_sequence + [a], env.s))
                        
        return [] # No solution found


def plot_animation(imgs):
    height, width = imgs[0].shape[:2]
    fig = plt.figure(figsize=(width/100, height/100), dpi=500)
    
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    im = ax.imshow(imgs[0])
    def init():
        im.set_data(imgs[0])
        return [im]
    def update(i):
        im.set_data(imgs[i])
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), init_func=init, blit=True)
    return ani

def solve_frozenlake(seed, size=6, p=0.8, is_slippery=True, saved_animation_path=None):
    """
    Solve the given frozen lake environment and save the animation
    """
    actions = get_shortest_action_path(seed, size, p, is_slippery)
    env = FrozenLakeEnv(size=size, p=p, is_slippery=is_slippery, seed=seed)
    env._reset(seed=seed)
    print(f"Found {len(actions)} actions: {actions}")
    imgs = []
    img_before_action = env.render('rgb_array')
    imgs.append(img_before_action)
    for action in actions:
        env._step(action)
        img_after_action = env._render('rgb_array')
        imgs.append(img_after_action)
    ani = plot_animation(imgs)
    ani.save(saved_animation_path)


if __name__ == "__main__":
    seed = 10003
    size = 6
    p = 0.8
    is_slippery = True
    saved_animation_path = f"frozenlake_{seed}.gif"
    solve_frozenlake(seed, size, p, is_slippery, saved_animation_path)