import sys
import os

# Add the directory containing feature_extractors to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import pickle
import gymnasium as gym
import tempfile

# imitation imports
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

# stable-baselines imports
import datetime
import os 
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from minigrid.wrappers import FlatObsWrapper
import feature_extractors.unlockpickup as fe
from stable_baselines3.common.evaluation import evaluate_policy

rng = np.random.default_rng(0)
os.makedirs("models/P2/IL", exist_ok=True)

# initialize the environment and the learner
env = gym.make("MiniGrid-UnlockPickup-v0")
# wrap the environment
env = FlatObsWrapper(env)
env = DummyVecEnv([lambda: env])


# initialize the learner
# load feature extractor
policy_kwargs = dict(
    features_extractor_class=fe.get("FlatObsFeatureExtractor"), # use default params
)

learner = PPO(env=env, policy="MlpPolicy", policy_kwargs=policy_kwargs, verbose=0)

# initialize from my best checkpoint so far
learner.load("models/P2/BEST/model.zip")


# load transitions
with open("imitation_learning/trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)

# initialize the BC trainer
today = datetime.date.today().strftime("%Y%m%d_%H%M%S")
wandb.init(
    project="A5-P2-IL",
    entity="jh216",
    name=f"GAIL-{today}",
)

reward_net = BasicRewardNet(
    env.observation_space,
    env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=trajectories,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

gail_trainer.train(50000,
                   callback=lambda r: wandb.log({"reward": r}),)
rewards, _ = evaluate_policy(learner, env, 100)
print("Rewards:", rewards)

# save the policy
learner.save("models/P2/IL/P2-GAIL.zip")
