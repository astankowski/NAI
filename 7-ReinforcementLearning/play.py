"""
Agent Visualization & Evaluation Module

This script loads a pre-trained DQN agent and runs a live simulation
render to evaluate the agent's performancey.

Authors:
- Aleksander Stankowski (s27549)
- Daniel Bieliński (s27292)

Features:
    - Loads trained model weights.
    - Configures the environment for human-friendly rendering.
    - Overrides internal environment parameters to increase difficulty (traffic density).

Usage:
    Ensure 'my_highway_agent_opt.zip' exists in the directory before running.
    python play.py
"""

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import sys

# Visualization Configuration

MODEL_PATH = "./my_highway_agent_opt"

# Custom environment configuration for the demo run
DEMO_CONFIG = {
    "simulation_frequency": 15,      # Higher frequency for smoother physics
    "policy_frequency": 5,           # How often the agent takes action
    "duration": 120,                 # Episode duration in seconds
    "vehicles_count": 40,            # High traffic density
    "screen_width": 1200,            # Resolution Width
    "screen_height": 600,            # Resolution Height
    "centering_position": [0.3, 0.5],# Camera focus
    "scaling": 5.5,                  # Zoom level
    "show_trajectories": False       # Disable debug lines
}

# Execution Logic

def watch_agent():
    """
    Loads the trained model and visualizes its behavior in the environment.

    Steps:
    1. Loads the .zip model file.
    2. Initializes 'highway-fast-v0' with 'human' render mode.
    3. Injects custom configuration using .unwrapped.configure() to bypass Gym wrappers.
    4. Runs the simulation loop until manually interrupted.

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    print(f"Loading Model: {MODEL_PATH}")
    
    try:
        model = DQN.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}.zip' not found")
        sys.exit(1)

    # Initialize Environment
    env = gym.make("highway-fast-v0", render_mode='human')
    
    env.unwrapped.configure(DEMO_CONFIG)

    print("Starting Simulation")
    
    try:
        while True:
            # Reset environment for a new episode
            obs, info = env.reset()
            done = truncated = False
            
            while not done and not truncated:
                # deterministic=True ensures the agent exploits its learned policy 
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, truncated, info = env.step(action)
                env.render()
                
    except KeyboardInterrupt:
        print("\nSimulation Stopped by User")
        env.close()

if __name__ == "__main__":
    watch_agent()
