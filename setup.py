"""
Setup script for drone RL project in VS Code Online
"""

# Create a virtual environment
import sys
import subprocess
import os

def setup_environment():
    """Set up the Python environment with all required packages"""
    print("Setting up environment for drone RL project...")
    
    # Check if running in VS Code Online/Codespaces
    in_codespace = os.environ.get('CODESPACES') == 'true'
    print(f"Running in Codespace/VS Code Online: {in_codespace}")
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Ensure pip is up to date
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install specific NumPy version to avoid compatibility issues
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.23.5"])
    
    # Install main dependencies
    packages = [
        "stable-baselines3",
        "tensorboard",
        "gymnasium==0.28.1",
        "pybullet",
        "matplotlib"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Install gym-pybullet-drones
    print("Installing gym-pybullet-drones...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/utiasDSL/gym-pybullet-drones.git@v1.0.0"
    ])
    
    # Test imports
    try:
        import numpy as np
        from stable_baselines3 import PPO
        from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
        print("✅ All imports successful!")
        print(f"NumPy version: {np.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = setup_environment()
    if success:
        print("Setup complete! You can now run the training script.")
    else:
        print("Setup encountered errors. Please check the error messages above.")