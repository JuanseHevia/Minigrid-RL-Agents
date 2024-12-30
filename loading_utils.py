import os
import json
import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device
from minigrid.wrappers import FlatObsWrapper
import feature_extractors.unlockpickup as extractor

def load_ppo_from_unzipped_folder(folder_path: str, default_env_id: str) -> PPO:
    # Load configuration data
    with open(os.path.join(folder_path, 'data'), 'r') as f:
        data = json.load(f)

    # Extract necessary information
    policy_class = "MlpPolicy"
    policy_kwargs = dict(
        features_extractor_class=getattr(extractor, data["feature_extractor"]),
        features_extractor_kwargs=data.get("extractor_params")
    )


    env_id = data.get("env_id", default_env_id)
    device = "cpu"

    # Load the environment
    env = gym.make(env_id)
    env = FlatObsWrapper(env)

    # Initialize the PPO model
    model = PPO(policy_class, env=env, policy_kwargs=policy_kwargs, device=device)
    
    # Load the policy state dictionary
    policy_path = os.path.join(folder_path, 'policy.pth')
    model.policy.load_state_dict(th.load(policy_path, map_location=device))

    # Load optimizer state if available
    optimizer_path = os.path.join(folder_path, 'policy.optimizer.pth')
    if os.path.exists(optimizer_path):
        model.policy.optimizer.load_state_dict(th.load(optimizer_path, map_location=device))

    return model