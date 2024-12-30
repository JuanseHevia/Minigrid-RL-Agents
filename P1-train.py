#!/usr/bin/env python
# coding: utf-8

# # Assignment 5, Problem 1
# 
# This is the starter code for Assignment 5, Problem 1.
# 
# In this assignment, you will solve increasingly challenging tasks from the [Minigrid benchmark](https://minigrid.farama.org/).

# In[1]:


# !pip install torch
# !pip install gymnasium==0.29.1
# !pip install minigrid==2.3.1
# !pip install stable-baselines3
# !pip install wandb
# !pip install tensorboard


# In[2]:


import gymnasium as gym
import minigrid
import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


# In[3]:


def compute_score(task, policy):
  num_episodes = 10
  cur_episode  = 0

  seed_by_episode = [42, 34, 50, 1, 9, 7, 43, 56, 90, 11]
  score_by_episode = np.zeros(num_episodes)

  while cur_episode < num_episodes:

    cumulative_reward = 0
    cur_seed = seed_by_episode[cur_episode]

    observation, info = task.reset(seed=cur_seed)
    done = False

    while not done:
      action = policy(observation)
      observation, reward, terminated, truncated, info = task.step(action)
      cumulative_reward += reward

      if terminated or truncated:
        done = True
        score_by_episode[cur_episode] = cumulative_reward
        cur_episode += 1

  score_mean = round(score_by_episode.mean(), 3)
  score_std  = round(score_by_episode.std(), 3)
  score_best = round(score_by_episode.max(), 3)

  print(f"Best score: {score_best}")
  print(f"Average score: {score_mean, score_std}")

  return score_by_episode


# ## Problem 1
# Solve the [Minigrid Unlock](https://minigrid.farama.org/environments/minigrid/UnlockEnv/) task.
# 
# This problem is worth 5 points.
# 
# ![](https://minigrid.farama.org/_images/UnlockEnv.gif)

# In[4]:


first_task = gym.make("MiniGrid-Unlock-v0")


# In[5]:


obs, _ = first_task.reset()
print("Observation keys:", obs.keys())
print("Image shape:", obs['image'].shape)
print("Direction:", obs["direction"])
print("Mission:", obs["mission"])
print("Action space:", first_task.action_space)


# # Training

# ## Define feature extractor
# Minigrid does not comply with SB3's off-the-shelf CnnPolicy

# In[6]:


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512,
                  normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# In[ ]:


######## PUT YOUR CODE HERE ########
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb

# setup config
_config = {
    "policy": "CnnPolicy",
    "env": "MiniGrid-Unlock-v0",
    "total_timesteps": 5e5,
    "features_dim": 128,
}

## initialize wandb project
run = wandb.init(project="COMP552-A5-minigrid",
           entity="jh216",
           config=_config,
           sync_tensorboard=True,
           monitor_gym=True)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=_config["features_dim"]),
)

env = gym.make("MiniGrid-Unlock-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
            tensorboard_log=f"runs/{run.id}")
model.learn(_config["total_timesteps"], callback=WandbCallback(
    model_save_path=f"models/P1/{run.id}",
    verbose=2,
))

model.save(f"models/{run.id}")
run.finish()
######## PUT YOUR CODE HERE ########


# In[27]:


def first_policy(observation):
  ######## PUT YOUR CODE HERE ########
  action, _ = model.predict(observation["image"])

  ######## PUT YOUR CODE HERE ########
  return action


# In[28]:


scores = compute_score(task=first_task, policy=first_policy)


# In[29]:


print(f"Points awarded: {10*max(max(scores) - 0.3, 0)}")

