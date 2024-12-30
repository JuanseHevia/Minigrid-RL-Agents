#!/usr/bin/env python3

from __future__ import annotations
import os 
import sys 
# include parent directory in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minigrid.wrappers as wrappers
import numpy as np
import custom_envs

import pygame
import pickle
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, FlatObsWrapper
from imitation.data.types import Trajectory
import gymnasium as gym

class ManualControl:
    def __init__(
        self,
        env: gym.Env,
        seed=None,
        save_path="trajectories.pkl",
        load_prev_traj=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.save_path = save_path
        
        # load previous trajectories
        if load_prev_traj is not None:
            print("Loading previous trajectories from {}".format(load_prev_traj))
            with open(load_prev_traj, 'rb') as f:
                self.trajectories = pickle.load(f)
            # drop last trajectory, which probably isn't complete
            self.trajectories = self.trajectories[:-1]
            # print number of trajectories loaded
            print("Loaded {} trajectories".format(len(self.trajectories)))
        else:
            self.trajectories = []

        self.reset_trajectory_storage()

    def reset_trajectory_storage(self):
        self.obs = []
        self.acts = []
        self.rews = []
        self.infos = []

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        self.ep_cum_reward = 0

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    self.save_trajectories()
                    self.closed = True
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        # add print statements to record the state of the door and the agent
        # print("Door is locked: ", self.env.grid.get(*self.env.door_position).is_locked)
        print("Agent carrying: ", self.env.carrying)
        print("Agent position: ", self.env.agent_pos)
        print("Door position: ", self.env.door_pos)

        self.ep_cum_reward += reward
        print(f"Episode cumulative reawrd: {self.ep_cum_reward:.2f}")
        
        # Record the transition
        self.obs.append(obs)
        self.acts.append(action)
        self.rews.append(reward)
        self.infos.append(info)

        if terminated or truncated:
            print("Episode ended!")
            # Append the trajectory
            trajectory = Trajectory(
                obs=np.array(self.obs),
                acts=np.array(self.acts),
                infos=np.array(self.infos),
                terminal=terminated,
            )
            self.trajectories.append(trajectory)
            self.reset_trajectory_storage()
            self.reset(self.seed)
            self.ep_cum_reward = 0
        else:
            self.env.render()

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self.obs.append(obs)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            self.save_trajectories()
            self.closed = True
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "d": Actions.pickup,  # Changed from "pageup"
            "f": Actions.drop,    # Changed from "pagedown"
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    def save_trajectories(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.trajectories, f)
        print(f"Saved {len(self.trajectories)} trajectories to {self.save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="UnlockPickup-RewardShaping-Simple-v1",
    )
    parser.add_argument(
        "--load_prev_traj",
        type=str,
        help="path to load previous trajectories",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="path to save the collected trajectories",
        default="trajectories.pkl",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        render_mode="human",
    )
    # wrap the environment
    env = FlatObsWrapper(env)
    
    manual_control = ManualControl(env, seed=args.seed, save_path=args.save_path, load_prev_traj=args.load_prev_traj)
    manual_control.start()