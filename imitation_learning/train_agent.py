import sys
import os

# Add the directory containing feature_extractors to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import pickle
import gymnasium as gym
import tempfile

# imitation imports
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.logger import WandbOutputFormat

# stable-baselines imports
import datetime
import os 
import wandb
import numpy as np
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper
import feature_extractors.unlockpickup as fe

rng = np.random.default_rng(0)
os.makedirs("models/P2/IL", exist_ok=True)

# initialize the environment and the learner
env = gym.make("MiniGrid-UnlockPickup-v0")
# wrap the environment
env = FlatObsWrapper(env)


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
    transitions = pickle.load(f)

# initialize the BC trainer
today = datetime.date.today().strftime("%Y%m%d_%H%M%S")
wandb.init(
    project="A5-P2-IL",
    entity="jh216",
    name=f"BC-{today}",
)


bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
    custom_logger=WandbOutputFormat(),
)


bc_trainer.train(n_epochs=10000,
                 progress_bar=True
                 )


bc_trainer.save_policy(f"models/P2/IL/P2-BC-{today}.zip")