import os
import pprint
import datetime
import torch
import json
from wandb.integration.sb3 import WandbCallback
import custom_envs
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import minigrid.wrappers as wrappers
import minigrid
import wandb
from munch import Munch
from feature_extractors import unlockpickup
import argparse

CONFIG = {
    "track": True,
    "env": "UnlockPickup-RewardShaping-v0",
    "wrapper": "ImgObsWrapper",
    "total_timesteps": 10000,
    "feature_extractor": "ImageFeatureExtractor",
    "extractor_params": {
        "features_dim": 128,
        "linear_layer_size": [512, 256, 128],
        "linear_num_layers": 3
    },
    "batch_size": 64,
}

PRETRAIN_PATHS = {
    "ImgObsWrapper": "models/P2/RS_Simple_v1_ImgFE_BS128-20241208-20_32_44/policy.pth",
}


def train(config):
    """
    Run a full training
    """

    # get current date
    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d-%H_%M_%S")
    # append to config name
    config["name"] = config.name + "-" + date

    if config.track:
        run = wandb.init(project="A5-P3", config=config,
                         name=config.name,
                         sync_tensorboard=True,
                         monitor_gym=True
                         )

        callback_wandb = WandbCallback(
            model_save_path=f"{config.savedir}/{config.name}",
            verbose=2,
        )
        run_id = run.id
    else:
        callback_wandb = []
        run_id = "notracking"

    print("Using env: ", config.env)
    env = gym.make(config.env)
    if config.wrapper is not None:
        print(f"Using wrapper : {config.wrapper}")
        assert hasattr(
            wrappers, config.wrapper), f"Wrapper {config.wrapper} not found in minigrid.wrappers"

        _wrapper = getattr(wrappers, config.wrapper)
        env = _wrapper(env)

    extractor = unlockpickup.get(config.feature_extractor)

    assert "extractor_params" in config, '''No feature extractor parameters found in the configuration. Add 'extractor_params':
                                             `params_dict` to the configuration dict.'''
    assert unlockpickup.validate_feature_extractor_args(
        config.extractor_params), "Invalid feature extractor parameters."

    policy_kwargs = dict(
        features_extractor_class=extractor,
        features_extractor_kwargs=config.extractor_params
    )

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0,
                batch_size=config.batch_size,
                tensorboard_log=f"runs/{run_id}")

    # load P1 model policy weights
    if config.load_pretrained:
        print(f"Loading pretrained policy: {PRETRAIN_PATHS[config.wrapper]}")
        model.policy.load_state_dict(
            torch.load(PRETRAIN_PATHS[config.wrapper]))

    if config.load_pretrained_path is not None:
        model.load(config.load_pretrained_path)

    model.learn(total_timesteps=CONFIG.total_timesteps,
                progress_bar=True,
                callback=callback_wandb)

    model.save(os.path.join(config.savedir, config.name))
    # save configuration as JSON
    if config.track:
        with open(os.path.join(config.savedir, f"{config.name}_config.json"), "w") as f:
            json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on a custom environment")
    parser.add_argument("--name", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--env", type=str, default="BlockedUnlockPickup-RewardShaping-v0",
                        required=False, help="Name of the environment")
    parser.add_argument("--track", type=lambda x: (str(x).lower() == 'true'),
                        default=False, required=False, help="Whether to track the training with wandb")
    parser.add_argument("--load_pretrained_path", type=str, default=None,
                        required=False, help="Path to the P1 model to load")
    parser.add_argument("--load_pretrained", type=lambda x: (str(x).lower() == 'true'),
                        default=True, required=False, help="Whether to load the pretrained policy weights")
    # add a savedir for P3
    parser.add_argument("--savedir", type=str, default="models/P3",
                        required=False, help="Directory to save the model")

    for key in CONFIG.keys():
        if key not in ["name", "env", "track", "extractor_params"]:
            parser.add_argument(
                f"--{key}", type=type(CONFIG[key]), default=CONFIG[key], required=False, help=f"Value for {key}")

    # add arguments for the extractor params
    for key in CONFIG["extractor_params"].keys():
        parser.add_argument(f"--extractor_params_{key}",
                            type=type(CONFIG["extractor_params"][key]),
                            default=CONFIG["extractor_params"][key], required=False,
                            help=f"Value for {key}")

    args = parser.parse_args()
    # if track is string, convert to boolean
    if isinstance(args.track, str):
        args.track = args.track.lower() == "true"

    # update CONFIG with args
    CONFIG["savedir"] = args.savedir
    CONFIG["load_pretrained_path"] = args.load_pretrained_path
    CONFIG["load_pretrained"] = args.load_pretrained
    CONFIG["name"] = args.name
    for key in CONFIG.keys():
        if key != "extractor_params":
            CONFIG[key] = getattr(args, key)

    # update extractor params
    for key in CONFIG["extractor_params"].keys():
        CONFIG["extractor_params"][key] = getattr(
            args, f"extractor_params_{key}")

    print("Using CONFIG: ")
    pprint.pprint(CONFIG)
    CONFIG = Munch(CONFIG)
    train(CONFIG)
