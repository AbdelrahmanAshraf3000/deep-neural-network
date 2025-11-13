# train.py
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import time
import numpy as np
from DiscretizedPendulumWrapper import DiscretizedPendulumWrapper
import os
# Import the classes we just defined
from model import QNetwork
from replay_memory import ReplayMemory, Transition

# Import wandb
import wandb # ## WANDB ##

# Set up device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def select_action(state, policy_net, n_actions, epsilon):
    """Selects an action using an epsilon-greedy policy."""
    sample = random.random()
    if sample > epsilon:
        # Exploitation: Choose the best action from the Q-network
        with torch.no_grad():
            # .max(1) returns (values, indices)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Exploration: Choose a random action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net, target_net, optimizer, config):
    """Performs one step of optimization on the policy network."""
    if len(memory) < config.BATCH_SIZE:
        return None  # Not enough samples

    transitions = memory.sample(config.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Prepare batches
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)  # shape: (BATCH_SIZE,)
    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state),
                                device=device, dtype=torch.bool)

    # Build next_state_batch only if any non-final
    if non_final_mask.any():
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
    else:
        next_state_batch = None

    # Compute current Q values
    state_action_values = policy_net(state_batch).gather(1, action_batch)  # shape: (BATCH_SIZE, 1)

    # Compute V(s_{t+1})
    next_state_values = torch.zeros(config.BATCH_SIZE, device=device)

    if next_state_batch is not None:
        if config.USE_DDQN:
            # DDQN: action selection by policy_net, evaluation by target_net
            with torch.no_grad():
                best_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                next_vals = target_net(next_state_batch).gather(1, best_actions).squeeze(1)
                next_state_values[non_final_mask] = next_vals
        else:
            with torch.no_grad():
                next_vals = target_net(next_state_batch).max(1)[0]
                next_state_values[non_final_mask] = next_vals

    # TD target
    expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

    # Loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()



def train(config):
    """Main training loop."""
    
    # ## WANDB ## Initialize a new run
    run = wandb.init(
        project="rl-classic-control", 
        config=config,
        name=f"{config['ENV_NAME']}_{'DDQN' if config['USE_DDQN'] else 'DQN'}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Get configuration from wandb object
    config = wandb.config


    if config.ENV_NAME == "Pendulum-v1":
        env = DiscretizedPendulumWrapper(gym.make("Pendulum-v1",render_mode="rgb_array"))
    else:
        env = gym.make(config.ENV_NAME,render_mode="rgb_array")

    
    # Get state and action space dimensions
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize networks
    policy_net = QNetwork(n_observations, n_actions).to(device)
    target_net = QNetwork(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Set target network to evaluation mode

    # Initialize optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=config.LR, amsgrad=True)
    
    # Initialize replay memory
    memory = ReplayMemory(config.MEMORY_SIZE)

    # Initialize epsilon for epsilon-greedy policy
    epsilon = config.EPS_START
    
    
    all_episode_rewards = [] # For logging

    print(f"Starting training for {config.NUM_EPISODES} episodes...")

    total_steps = 0

    # Main training loop
    for i_episode in range(config.NUM_EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        episode_reward = 0
        episode_steps = 0
        episode_loss = []

        while True: # Limit episode length
            episode_steps += 1
            total_steps += 1
            
            # Linearly decay epsilon
            epsilon = max(config.EPS_END, config.EPS_START - (config.EPS_START - config.EPS_END) * (total_steps / config.EPS_DECAY))
            
            # Select and perform an action
            action = select_action(state, policy_net, n_actions, epsilon)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            episode_reward += reward
            done = terminated or truncated or (episode_steps >= config.MAX_STEPS_PER_EPISODE)
            

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model(memory, policy_net, target_net, optimizer, config)
            if loss is not None:
                episode_loss.append(loss)

            # Soft update of the target network's weights
            # θ_target = τ*θ_policy + (1 - τ)*θ_target
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*config.TAU + target_net_state_dict[key]*(1-config.TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
        
        all_episode_rewards.append(episode_reward)
        avg_reward = np.mean(all_episode_rewards[-100:]) # Moving avg of last 100 episodes
        avg_loss = np.mean(episode_loss) if episode_loss else 0

        # ## WANDB ## Log metrics
        wandb.log({
            "episode": i_episode,
            "episode_reward": episode_reward,
            "avg_reward_100": avg_reward,
            "avg_loss": avg_loss,
            "epsilon": epsilon,
            "total_steps": episode_steps
        })

        if i_episode % 20 == 0:
            print(f"Episode {i_episode}: Reward: {episode_reward:.2f}, Avg Reward (100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Steps: {episode_steps}, Avg Loss: {avg_loss:.4f}")
            
    print("Training complete.")
    
    # Save the trained model
    model_path = f"./{run.name}_model.pth"
    torch.save(policy_net.state_dict(), model_path)
    # ## WANDB ## Save model to wandb
    #wandb.save(model_path)
    
    # Close environment
    env.close()
    
    # Run tests
    test_agent(policy_net, config, run)
    
    # ## WANDB ## Finish the run
    run.finish()


### 📊 Step 7 & 9: Test Runs & Video Recording
#This function runs the trained agent and logs the test results to the *same* W&B run. It also handles recording a video for one of the test runs.

def test_agent(trained_policy_net, config, run):
    """Run the trained agent 100 times and record a video."""
    print("Starting testing...")
    
    # --- Video Recording Setup (Step 9) ---
    # Create a new environment, wrapping it with RecordVideo
    # This will record one video of the first test episode
    # --- Video Recording Setup ---

    # --- Ensure video directory exists ---
    os.makedirs(f"./videos/{run.name}", exist_ok=True)

    # --- Create and wrap environment ---
    if config.ENV_NAME == "Pendulum-v1":
        base_env = DiscretizedPendulumWrapper(
            gym.make("Pendulum-v1", render_mode="rgb_array")
        )
    else:
        base_env = gym.make(config.ENV_NAME, render_mode="rgb_array")

    video_env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=f"./videos/{run.name}",
        episode_trigger=lambda e: e % 10 == 0  # record every 10th episode
    )

    n_actions = video_env.action_space.n
    test_episode_durations = []
    test_rewards = []

    for i in range(100):

        
        state, info = video_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        terminated = False
        truncated = False
        duration = 0
        episode_reward = 0.0
        
        while True:
            duration += 1
            # Run the agent with epsilon = 0 (pure exploitation)
            action = select_action(state, trained_policy_net, n_actions, epsilon=0.0)
            observation, reward, terminated, truncated, _ = video_env.step(action.item())
            episode_reward += reward   
            done = terminated or truncated or (duration >= config.MAX_STEPS_PER_EPISODE)   
            if done:
                break
            else:
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        
        test_episode_durations.append(duration)
        test_rewards.append(episode_reward)
        wandb.log({"test_episode_duration": duration})
        print(f"Test Episode {i}: Duration: {duration}, Reward: {episode_reward:.2f}")
        
    # Close environments
    video_env.close()
    
    # Log results to W&B
    avg_duration = np.mean(test_episode_durations)
    std_duration = np.std(test_episode_durations)
    avg_reward = np.mean(test_rewards)
    print(f"Testing complete. Avg Reward over 100 episodes: {avg_reward:.2f}")
    print(f"Testing complete. Avg Duration: {avg_duration:.2f} +/- {std_duration:.2f} ")

    # ## WANDB ## Log test statistics
    wandb.log({
        "test_avg_duration": avg_duration,
        "test_std_duration": std_duration,
        "test_avg_reward": avg_reward,
    })


    
    # ## WANDB ## Log the recorded video
    video_folder = f"./videos/{run.name}"

    # Try logging only existing files
    for e in range(0, 100, 10):  # every 10 episodes
        video_path = os.path.join(video_folder, f"rl-video-episode-{e}.mp4")
        if os.path.exists(video_path):
            wandb.log({
                f"test_video_episode_{e}": wandb.Video(
                    video_path, 
                    caption=f"Test Episode {e}", 
                    format="mp4"
                )
            })
        else:
            print(f"[WARN] Video not found for episode {e}: {video_path}")

    
    

if __name__ == "__main__":
    wandb.login()  # optional, for W&B logging

    # --- Base configuration template ---
    base_config = {
        "LR": 1e-4,
        "BATCH_SIZE": 128,
        "MEMORY_SIZE": 10000,
        "GAMMA": 0.99,
        "EPS_START": 1.0,
        "EPS_END": 0.05,
        "EPS_DECAY": 20000,
        "TAU": 0.005,
        "NUM_EPISODES": 600,
        "MAX_STEPS_PER_EPISODE": 1000,
    }

    # ------------------------------------------------------------------
    #  CartPole-v1 — DQN    
    # ------------------------------------------------------------------
    config_cartpole_dqn = base_config.copy()
    config_cartpole_dqn.update({
        "ENV_NAME": "CartPole-v1",
        "USE_DDQN": False,
        "LR": 1e-3,
        "EPS_DECAY": 5000,
        "NUM_EPISODES": 400,
        "MAX_STEPS_PER_EPISODE": 500,
    })
    train(config_cartpole_dqn)
    #test was working fine

    # # ------------------------------------------------------------------
    # #  CartPole-v1 — DDQN
    # # ------------------------------------------------------------------
    # config_cartpole_ddqn = base_config.copy()
    # config_cartpole_ddqn.update({
    #     "ENV_NAME": "CartPole-v1",
    #     "USE_DDQN": True,
    #     "LR": 5e-4,
    #     "EPS_DECAY": 8000,
    #     "NUM_EPISODES": 400,
    #     "MAX_STEPS_PER_EPISODE": 500,
    # })
    # train(config_cartpole_ddqn)
    # #test was so bad

    # # ------------------------------------------------------------------
    # #  Acrobot-v1 — DQN
    # # ------------------------------------------------------------------
    # config_acrobot_dqn = base_config.copy()
    # config_acrobot_dqn.update({
    #     "ENV_NAME": "Acrobot-v1",
    #     "USE_DDQN": False,
    #     "LR": 1e-3,
    #     "EPS_DECAY": 25000,
    #     "NUM_EPISODES": 1200,
    #     "MAX_STEPS_PER_EPISODE": 500,
    # })
    # train(config_acrobot_dqn)
    # #test was good

    # # ------------------------------------------------------------------
    # #  Acrobot-v1 — DDQN
    # # ------------------------------------------------------------------
    # config_acrobot_ddqn = base_config.copy()
    # config_acrobot_ddqn.update({
    #     "ENV_NAME": "Acrobot-v1",
    #     "USE_DDQN": True,
    #     "LR": 1e-3,
    #     "EPS_DECAY": 30000,
    #     "NUM_EPISODES": 1500,
    #     "MAX_STEPS_PER_EPISODE": 500,
    # })
    # train(config_acrobot_ddqn)
    # #test was better

    # # ------------------------------------------------------------------
    # #  MountainCar-v0 — DQN
    # # ------------------------------------------------------------------
    # config_mountaincar_dqn = base_config.copy()
    # config_mountaincar_dqn.update({
    #     "ENV_NAME": "MountainCar-v0",
    #     "USE_DDQN": False,
    #     "LR": 1e-3,
    #     "EPS_DECAY": 60000,              # slower decay = more exploration
    #     "NUM_EPISODES": 2500,
    #     "MAX_STEPS_PER_EPISODE": 200,
    #     "BATCH_SIZE": 256,
    # })
    # train(config_mountaincar_dqn)
    # # test was good

    # # ------------------------------------------------------------------
    # #  MountainCar-v0 — DDQN
    # # ------------------------------------------------------------------
    # config_mountaincar_ddqn = base_config.copy()
    # config_mountaincar_ddqn.update({
    #     "ENV_NAME": "MountainCar-v0",
    #     "USE_DDQN": True,
    #     "LR": 1e-3,
    #     "EPS_DECAY": 80000,
    #     "NUM_EPISODES": 3000,
    #     "MAX_STEPS_PER_EPISODE": 200,
    #     "BATCH_SIZE": 256,
    # })
    # train(config_mountaincar_ddqn)
    # #not enough time to test

    # # ------------------------------------------------------------------
    # # Pendulum-v1 — DQN (discretized)
    # # ------------------------------------------------------------------
    # config_pendulum_dqn = base_config.copy()
    # config_pendulum_dqn.update({
    #     "ENV_NAME": "Pendulum-v1",
    #     "USE_DDQN": False,
    #     "LR": 5e-4,
    #     "EPS_DECAY": 40000,
    #     "NUM_EPISODES": 200,
    #     "MAX_STEPS_PER_EPISODE": 200,
    # })
    # train(config_pendulum_dqn)

    # ------------------------------------------------------------------
    # # Pendulum-v1 — DDQN (discretized)
    # # ------------------------------------------------------------------
    # config_pendulum_ddqn = base_config.copy()
    # config_pendulum_ddqn.update({
    #     "ENV_NAME": "Pendulum-v1",
    #     "USE_DDQN": True,
    #     "LR": 5e-4,
    #     "EPS_DECAY": 50000,
    #     "NUM_EPISODES": 1500,
    #     "MAX_STEPS_PER_EPISODE": 200,
    # })
    # train(config_pendulum_ddqn)

