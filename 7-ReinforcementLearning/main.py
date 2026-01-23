"""
Reinforcement Learning Autonomous Driving Agent

This script trains a Deep Q-Network (DQN) agent to drive autonomously in a highway environment.

Problem:
    The goal is to navigate a vehicle in a dense traffic environment ('highway-fast-v0'),
    maximizing speed and distance traveled while avoiding collisions.

Frameworks & Libraries:
    - Gymnasium (Environment interface)
    - Highway-env (Simulation logic)
    - Stable Baselines3 (RL Algorithms: DQN)
    - PyTorch (Deep Learning backend)

Authors:
- Aleksander Stankowski (s27549)
- Daniel Bieliński (s27292)

Environment Setup:
    1. Install dependencies:
        pip install gymnasium highway-env stable-baselines3 torch shimmy
    2. Run the training:
        python main.py
"""

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import os
import time

# Configuration and parameters

MODEL_NAME = "my_highway_agent_opt"
TRAINING_STEPS = 500_000
N_CPU_PROCESSES = 16  # Limit parallel environments

DQN_PARAMS = {
    "policy": "MlpPolicy",
    "policy_kwargs": dict(net_arch=[512, 512]),  # Large network for complex decision making
    "learning_rate": 1e-4,                       # Lower LR for stability with large batches
    "buffer_size": 1_000_000,                    # Replay buffer
    "learning_starts": 5000,                     # Warmup steps
    "batch_size": 2048,                          # Batch size
    "gamma": 0.8,                                # Discount factor (0.8 = focus on near-future)
    "train_freq": 4,                             # Update network every 4 steps
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "verbose": 1,
    "device": "cuda"
}

# Training Functions

def train_optimized():
    """
    Configures and executes the training pipeline using Vectorized Environments.

    This function detects hardware capabilities, initializes multiple parallel 
    instances of the highway environment, and trains the DQN agent.

    Process:
    1. Hardware Detection (CPU cores/GPU).
    2. Environment Vectorization (SubprocVecEnv).
    3. Model Initialization (DQN with custom params).
    4. Training Loop.
    5. Model Persistence (Saving to .zip).

    Returns:
        None: Saves the trained model to disk as 'my_highway_agent_opt.zip'.
    """
    
    # Hardware Configuration
    available_cpu = os.cpu_count()
    n_envs = min(N_CPU_PROCESSES, available_cpu)
    
    print(f"\nHardware Detection")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CPU Cores: {available_cpu} (Using {n_envs} parallel environments)")

    # Environment Setup
    # make_vec_env creates a wrapper that manages multiple env instances simultaneously
    env = make_vec_env(
        env_id="highway-fast-v0", 
        n_envs=n_envs, 
        seed=0, 
        vec_env_cls=SubprocVecEnv  # Ensures each env runs in a separate process
    )

    # Model Initialization
    print(f"\nInitializing DQN Model")
    model = DQN(env=env, **DQN_PARAMS)

    # Training Loop
    print(f"\nStarting Training for {TRAINING_STEPS} steps")
    start_time = time.time()
    
    model.learn(total_timesteps=TRAINING_STEPS, progress_bar=True)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training Finished in {duration:.2f} seconds ({duration/60:.1f} min)")
    
    # Save Model
    model.save(MODEL_NAME)
    print(f"Model saved to: {MODEL_NAME}.zip")

if __name__ == "__main__":
    train_optimized()
