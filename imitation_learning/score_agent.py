import os 
import sys

import numpy as np 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import gymnasium as gym
import feature_extractors.unlockpickup as fe
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from imitation.algorithms import bc
from stable_baselines3.common.monitor import Monitor


# initialize the environment and the learner
env = gym.make("MiniGrid-UnlockPickup-v0")
# wrap the environment
env = FlatObsWrapper(env)
env = Monitor(env)


# initialize the learner
# load feature extractor
policy_kwargs = dict(
    features_extractor_class=fe.get("FlatObsFeatureExtractor"), # use default params
)

model = PPO(env=env, policy="MlpPolicy", policy_kwargs=policy_kwargs, verbose=0)
# load the BC policy
# model = bc.reconstruct_policy("models/P2/IL/P2-BC-20241208_000000.zip")
model.load("models/P2/IL/P2-GAIL.zip")

def compute_score(task, policy):
    num_episodes = 10
    cur_episode = 0

    seed_by_episode = [42, 34, 50, 1, 9, 7, 43, 56, 90, 11]
    # seed_by_episode = np.random.randint(0, 100, num_episodes)
    score_by_episode = np.zeros(num_episodes)

    while cur_episode < num_episodes:

        cumulative_reward = 0
        cur_seed = seed_by_episode[cur_episode]

        observation, info = task.reset(seed=int(cur_seed))
        done = False

        while not done:
            action = policy(observation)
            observation, reward, terminated, truncated, info = task.step(
                action)
            cumulative_reward += reward

            if terminated or truncated:
                done = True
                score_by_episode[cur_episode] = cumulative_reward
                cur_episode += 1

    score_mean = round(score_by_episode.mean(), 3)
    score_std = round(score_by_episode.std(), 3)
    score_best = round(score_by_episode.max(), 3)

    print(f"Best score: {score_best}")
    print(f"Average score: {score_mean, score_std}")

    return score_by_episode

policy = lambda obs: model.predict(obs)[0] # get action from model

compute_score(env, policy)