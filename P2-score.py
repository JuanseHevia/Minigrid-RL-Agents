import gymnasium as gym
import minigrid
import os
import minigrid.wrappers as wrappers
import numpy as np
from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper, DictObservationSpaceWrapper
from stable_baselines3 import PPO
import torch
import feature_extractors.unlockpickup as fe
import argparse
import json


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

def compute_score(task, policy):
    num_episodes = 10
    cur_episode = 0

    seed_by_episode = [42, 34, 50, 1, 9, 7, 43, 56, 90, 11]
    # seed_by_episode = np.random.randint(0, 1000, num_episodes)
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


def load_model_get_policy(model_path: str, config):

    _env = gym.make("MiniGrid-UnlockPickup-v0")
    _env = getattr(wrappers, config['wrapper'])(_env)

    # load the feacture extractor
    policy_kwargs = dict(
        features_extractor_class=fe.get(config['feature_extractor']),
        features_extractor_kwargs=config['extractor_params']
    )
    model = PPO("MlpPolicy", _env, policy_kwargs=policy_kwargs, verbose=1)
    
    model.policy.load_state_dict(torch.load(model_path))

    def policy(observation):
        ######## PUT YOUR CODE HERE ########
        action, _ = model.predict(observation)
        ######## PUT YOUR CODE HERE ########
        return action

    return policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute the score of a policy')
    parser.add_argument('--model_path', type=str,
                        required=True, help='Path to the model weights')
    parser.add_argument('--config_path', type=str,
                        required=False, help='Path to the configuration file')

    args = parser.parse_args()

    # if no config path provided, default to model path filename without extension and append _config.json
    if args.config_path is None:
        args.config_path = os.path.join(os.path.dirname(args.model_path),
                                        os.path.basename(args.model_path).split('.')[0] + '_config.json')

    # load the configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    policy = load_model_get_policy(args.model_path, config)
    task = gym.make("MiniGrid-UnlockPickup-v0")
    task = getattr(wrappers, config['wrapper'])(task)

    compute_score(task, policy)
