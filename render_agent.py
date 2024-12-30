import json
import os
import gymnasium as gym
import imageio
import numpy as np
import minigrid.wrappers as wrappers
from stable_baselines3 import PPO
import argparse
import feature_extractors.unlockpickup as fe

def load_model(model_path, env, config):

    # load the feacture extractor
    policy_kwargs = dict(
        features_extractor_class=fe.get(config['feature_extractor']),
        features_extractor_kwargs=config['extractor_params']
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.load(model_path)
    return model

def create_gif(env, model, num_episodes=5, max_steps=1000):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        frames = []
        cum_reward = 0
        for step in range(max_steps):
            frames.append(env.render())
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cum_reward += reward
            if done:
                break
        # gif_filename = os.path.join(gif_path, f'episode_{episode + 1}-CR{cum_reward}.gif')
        # imageio.mimsave(gif_filename, frames, fps=30)
        # print(f'GIF saved: {gif_filename}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render agent performance and save as GIF')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model weights')
    # parser.add_argument('--gif_path', type=str, required=True, help='Path to save the GIFs')
    parser.add_argument('--config_path', type=str, required=False, help='Path to the configuration file')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to record')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of steps per episode')

    args = parser.parse_args()

    # if config path not provided, default to model path filename without extension and append _config.json
    if args.config_path is None:
        args.config_path = os.path.join(os.path.dirname(args.model_path), os.path.basename(args.model_path).split('.')[0] + '_config.json')
    print("Configuration path: ", args.config_path)

    # if not os.path.exists(args.gif_path):
    #     os.makedirs(args.gif_path, exist_ok=True)

    # load the configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # NOTE: use the evaluation environment, not the training one!!
    _env = gym.make("MiniGrid-UnlockPickup-v0", render_mode='human')
    _env = getattr(wrappers, config['wrapper'])(_env)

    model = load_model(model_path=args.model_path, env=_env, config=config)
    create_gif(_env, model, args.num_episodes, args.max_steps)