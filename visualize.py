"""
Visualization script for drone RL results in VS Code Online
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
import time

class DroneVisualizer:
    """Class to visualize trained drone models"""
    
    def __init__(self, model_path="models/hover_final"):
        """Initialize with path to a trained model"""
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found.")
            self.model = None
        else:
            self.model = PPO.load(model_path)
            print(f"Loaded model from {model_path}")
    
    def create_hover_env(self, gui=True):
        """Create a hover environment with optional GUI"""
        env = HoverAviary(
            drone_model="cf2x",
            initial_xyzs=np.array([[0, 0, 0.5]]),
            act=ActionType.RPM,
            obs=ObservationType.KIN,
            freq=50,
            gui=gui,  # Set to True to see the simulation (if supported)
            record=True  # Record frames for visualization
        )
        return env
    
    def record_trajectory(self, steps=200):
        """Record drone trajectory for visualization"""
        if self.model is None:
            print("No model loaded. Please initialize with a valid model path.")
            return None
        
        # Create environment without GUI for faster recording
        env = self.create_hover_env(gui=False)
        
        trajectory = []
        action_history = []
        reward_history = []
        
        # Reset environment
        obs, info = env.reset()
        
        # Run episode
        for i in range(steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get drone state
            drone_state = env._getDroneStateVector(0)
            position = drone_state[0:3]  # x, y, z
            
            trajectory.append(position)
            action_history.append(action)
            reward_history.append(reward)
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Convert to numpy arrays
        trajectory = np.array(trajectory)
        action_history = np.array(action_history)
        reward_history = np.array(reward_history)
        
        return {
            'trajectory': trajectory,
            'actions': action_history, 
            'rewards': reward_history
        }
    
    def plot_3d_trajectory(self, data=None):
        """Plot 3D trajectory of the drone"""
        if data is None:
            data = self.record_trajectory()
            if data is None:
                return
        
        trajectory = data['trajectory']
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2)
        ax.plot(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 'ro', markersize=10, label='End')
        
        # Add target position for hover task
        target_position = [0, 0, 1.0]  # Default target for HoverAviary
        ax.plot([target_position[0]], [target_position[1]], [target_position[2]], 'ko', markersize=15, label='Target')
        
        # Add labels and legend
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('Drone Navigation Trajectory')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([
            trajectory[:, 0].max() - trajectory[:, 0].min(),
            trajectory[:, 1].max() - trajectory[:, 1].min(),
            trajectory[:, 2].max() - trajectory[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) * 0.5
        mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) * 0.5
        mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig('logs/trajectory_3d.png')
        plt.show()
    
    def plot_height_profile(self, data=None):
        """Plot height profile over time"""
        if data is None:
            data = self.record_trajectory()
            if data is None:
                return
                
        trajectory = data['trajectory']
        rewards = data['rewards']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot height over time
        time_steps = np.arange(len(trajectory))
        ax1.plot(time_steps, trajectory[:, 2], 'b-', linewidth=2)
        ax1.axhline(y=1.0, color='r', linestyle='--', label='Target height')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Drone Height Profile')
        ax1.legend()
        ax1.grid(True)
        
        # Plot rewards
        ax2.plot(time_steps, rewards, 'g-', linewidth=2)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Reward')
        ax2.set_title('Rewards per Step')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('logs/height_profile.png')
        plt.show()
    
    def create_performance_report(self, num_episodes=5, steps_per_episode=200):
        """Create a comprehensive performance report"""
        if self.model is None:
            print("No model loaded. Please initialize with a valid model path.")
            return
        
        print("Creating performance report...")
        print("-" * 50)
        
        # Track metrics across episodes
        all_rewards = []
        target_height = 1.0
        height_errors = []
        episode_lengths = []
        
        # Run multiple test episodes
        for episode in range(num_episodes):
            env = self.create_hover_env(gui=False)
            obs, info = env.reset()
            
            episode_reward = 0
            episode_height_errors = []
            
            for step in range(steps_per_episode):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track metrics
                episode_reward += reward
                height = obs[2]
                height_error = abs(height - target_height)
                episode_height_errors.append(height_error)
                
                if terminated or truncated:
                    break
            
            env.close()
            
            # Store episode metrics
            all_rewards.append(episode_reward)
            height_errors.append(np.mean(episode_height_errors))
            episode_lengths.append(step + 1)
            
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, " +
                  f"Avg Height Error = {np.mean(episode_height_errors):.4f}m, " +
                  f"Steps = {step+1}")
        
        # Print summary statistics
        print("-" * 50)
        print("Performance Summary:")
        print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        print(f"Average Height Error: {np.mean(height_errors):.4f}m ± {np.std(height_errors):.4f}m")
        print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
        print("-" * 50)
        
        # Create plots
        self.plot_3d_trajectory()
        self.plot_height_profile()

# Main execution
if __name__ == "__main__":
    # Check which models are available
    available_models = [f for f in os.listdir("models") if f.endswith("_final")]
    
    if not available_models:
        print("No trained models found in the 'models' directory.")
        print("Please train a model first or specify the correct model path.")
    else:
        print(f"Available models: {available_models}")
        # Use the first available model
        model_path = os.path.join("models", available_models[0])
        
        # Create visualizer and generate report
        visualizer = DroneVisualizer(model_path=model_path)
        visualizer.create_performance_report()