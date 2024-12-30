
# Reinforcement Learning Project - Minigrid Environments using PPO

This repository contains work completed for the course **COMP552 Reinforcement Learning** at Rice University. The goal of this project is to explore and implement various reinforcement learning algorithms in custom environments to solve open-ended tasks. The tasks are based on the Minigrid Unlock environment and progressively increase in complexity.

## Task Description

This project consists of three tasks:

### Task 1: Baseline Model Training
The first task involves solving the **Minigrid Unlock environment** using the **Proximal Policy Optimization (PPO)** algorithm.  
- A custom feature extractor is employed to process observations in the environment.
- The model is trained for 500,000 timesteps and evaluated over 10 episodes to measure performance.

### Task 2: Incorporating Reward Shaping and Pretrained Models
Building on Task 1, this task:
- Uses a modified environment (`UnlockPickup-RewardShaping-v0`) with **reward shaping** to provide additional incentives for desired behaviors.
- Allows for the use of pretrained weights from Task 1 to accelerate training and improve performance.
- Tracks the training process using **Weights & Biases (wandb)** for better visualization and analysis.

### Task 3: Image-Based Observations and Advanced Feature Extraction
The third task adds complexity:
- The environment is wrapped using the `ImgObsWrapper` to provide image-based observations.
- A custom image feature extractor is used to handle these observations effectively.
- Advanced reward shaping techniques are applied to navigate the more challenging environment configurations.

---

## Methodology

### Custom Environments
The **custom environments** were implemented under the `custom_envs` module, leveraging **reward shaping** to guide agent behavior. Reward shaping was designed to provide intermediate rewards for achieving subgoals, improving training efficiency and stability.

### Training Process
- **Feature Extraction**: Custom feature extractors were developed to adapt to different observation spaces (`FlatObsWrapper` and `ImgObsWrapper`).
- **Algorithm**: All tasks utilized the PPO algorithm with specific configurations tailored for each task.
- **Tracking**: Training was tracked using **wandb** for progress visualization and model checkpointing.

---

## Replication

### Training Models
To replicate the results for each task, follow these steps:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Task 1**:
   ```bash
   python P1-train.py --name Task1-Model
   ```

3. **Train Task 2**:
   ```bash
   python P2-train.py --name Task2-Model --load_pretrained True
   ```

4. **Train Task 3**:
   ```bash
   python P3-train.py --name Task3-Model --wrapper ImgObsWrapper
   ```

### Evaluation
To evaluate a trained model:
1. Use the following command, replacing `MODEL_PATH` with the path to the saved model:
   ```bash
   python evaluate.py --model_path MODEL_PATH --episodes 10
   ```
2. Evaluation results (reward and success rate) will be logged to the console and optionally tracked with wandb if enabled.

---

## Repository Structure

```
.
├── custom_envs/          # Custom environment implementations with reward shaping
├── feature_extractors/   # Feature extractors for processing observations
├── models/               # Directory to save trained models
├── P1-train.py           # Script for Task 1 training
├── P2-train.py           # Script for Task 2 training
├── P3-train.py           # Script for Task 3 training
├── evaluate.py           # Script for model evaluation
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

---

## Contact
For questions or feedback, feel free to reach out!
