"""
Drone RL training script for VS Code Online/Codespaces
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# Import gym-pybullet-drones environment
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync

# Make sure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

class DroneTrainer:
    """Class to manage drone RL training"""
    
    def __init__(self, task="hover", gui=False, training_steps=100000):
        """Initialize trainer with task type and parameters"""
        self.task = task
        self.gui = gui
        self.training_steps = training_steps
        self.model = None
        
        # Create environment based on task
        if task == "hover":
            self.env = self._create_hover_env()
        else:
            raise ValueError(f"Unknown task: {task}. Supported tasks: 'hover'")
    
    def _create_hover_env(self):
        """Create a hover task environment"""
        env = HoverAviary(
            drone_model="cf2x",
            initial_xyzs=np.array([[0, 0, 0.5]]),
            act=ActionType.RPM,
            obs=ObservationType.KIN,
            freq=50,
            gui=self.gui,
            record=False
        )
        
        # Add monitoring
        env = Monitor(env, f"logs/train")
        return env
    
    def train(self, checkpoint_interval=10000):
        """Train the drone RL agent"""
        print(f"Starting {self.task} training for {self.training_steps} steps...")
        
        # Create a PPO agent
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log="logs/tb/",
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64
        )
        
        # Train in chunks with checkpoints
        steps_completed = 0
        while steps_completed < self.training_steps:
            steps_to_train = min(checkpoint_interval, self.training_steps - steps_completed)
            if steps_to_train <= 0:
                break
                
            print(f"Training for {steps_to_train} steps...")
            self.model.learn(total_timesteps=steps_to_train)
            
            steps_completed += steps_to_train
            print(f"Completed {steps_completed}/{self.training_steps} steps")
            
            # Save checkpoint
            self.model.save(f"models/{self.task}_checkpoint_{steps_completed}")
        
        # Save final model
        self.model.save(f"models/{self.task}_final")
        print("Training complete!")
        return self.model
    
    def test(self, model_path=None, steps=100):
        """Test trained model"""
        if model_path:
            self.model = PPO.load(model_path)
        elif self.model is None:
            raise ValueError("No model available. Either train first or provide a model_path.")
        
        print("\nTesting model performance...")
        
        # Reset environment
        env = self._create_hover_env()
        obs, info = env.reset()
        
        # Run test episode
        total_reward = 0
        target_height = 1.0  # Default target for HoverAviary
        
        print("Format: Step | Height | Target | Error | Reward")
        print("-" * 50)
        
        for i in range(steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get drone position
            height = obs[2]  # Z coordinate in observation
            height_error = abs(height - target_height)
            
            total_reward += reward
            
            # Print progress every 10 steps
            if i % 10 == 0:
                print(f"Step {i:3d} | Height: {height:.2f}m | Target: {target_height:.2f}m | Error: {height_error:.2f}m | Reward: {reward:.2f}")
            
            if terminated or truncated:
                print("Episode ended early")
                break
        
        env.close()
        print("-" * 50)
        print(f"Test complete. Total reward: {total_reward:.2f}")
        
        return total_reward
    
    def plot_training_progress(self):
        """Plot training progress from logs"""
        try:
            # Read monitor log
            from stable_baselines3.common.results_plotter import load_results, ts2xy
            log_path = "logs"
            
            # Plot training reward
            plt.figure(figsize=(10, 5))
            
            # Plot episodic reward
            x, y = ts2xy(load_results(log_path), 'timesteps')
            plt.plot(x, y, label='Reward per episode')
            
            # Add trend line
            from scipy.ndimage.filters import gaussian_filter1d
            y_smooth = gaussian_filter1d(y, sigma=2)
            plt.plot(x, y_smooth, label='Trend', linewidth=2)
            
            plt.xlabel('Timesteps')
            plt.ylabel('Reward')
            plt.title('Training Progress')
            plt.legend()
            plt.savefig(f'logs/{self.task}_progress.png')
            plt.show()
        except Exception as e:
            print(f"Error generating plot: {e}")

# Main execution
if __name__ == "__main__":
    # Create and train a drone hover task
    trainer = DroneTrainer(task="hover", gui=False, training_steps=50000)
    model = trainer.train(checkpoint_interval=10000)
    
    # Test the trained model
    trainer.test()
    
    # Plot training progress
    trainer.plot_training_progress()
    
    print("Drone RL project execution complete!")