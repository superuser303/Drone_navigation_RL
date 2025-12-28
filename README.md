# Drone Navigation with Reinforcement Learning

A reinforcement learning project for training drones to perform autonomous navigation tasks using the PPO (Proximal Policy Optimization) algorithm. This project focuses on the hover task, where a drone learns to maintain a stable position at a target height.

## Features

- **Reinforcement Learning Training**: Uses Stable Baselines3 PPO algorithm to train drone control policies
- **PyBullet Simulation**: Realistic drone physics simulation using the gym-pybullet-drones environment
- **Visualization Tools**: 3D trajectory plotting, height profiles, and performance analysis
- **Checkpointing**: Automatic model saving during training for recovery and evaluation
- **Performance Reporting**: Comprehensive evaluation metrics and visualization of trained models

## Project Structure

```
Drone_navigation_RL/
├── README.md              # Project documentation
├── setup.py               # Environment setup script
├── train.py               # Training script for RL agents
├── visualize.py           # Visualization and evaluation tools
├── logs/                  # Training logs and generated plots
└── models/                # Saved trained models
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/superuser303/Drone_navigation_RL.git
   cd Drone_navigation_RL
   ```

2. Run the setup script to install all dependencies:
   ```bash
   python setup.py
   ```

   This will install:
   - stable-baselines3 (RL framework)
   - gymnasium (environment interface)
   - pybullet (physics simulation)
   - gym-pybullet-drones (drone-specific environments)
   - tensorboard (logging and visualization)
   - matplotlib (plotting)

## Usage

### Training

To train a drone to hover at a target height:

```bash
python train.py
```

The training script will:
- Create a hover environment with a Crazyflie 2.0 drone
- Train a PPO agent for 50,000 timesteps
- Save checkpoints every 10,000 steps
- Generate training progress plots
- Save the final trained model

### Visualization and Evaluation

To visualize and evaluate a trained model:

```bash
python visualize.py
```

This will:
- Load the trained model from `models/hover_final`
- Generate a 3D trajectory plot
- Create height profile and reward plots
- Run performance evaluation across multiple episodes
- Display comprehensive metrics including average reward and height error

## Environment Details

- **Drone Model**: Crazyflie 2.0 (cf2x)
- **Task**: Hover at 1.0m height
- **Action Space**: RPM (rotations per minute) for 4 motors
- **Observation Space**: Kinematic state (position, velocity, orientation)
- **Simulation Frequency**: 50 Hz

## Dependencies

- numpy >= 1.26.0
- stable-baselines3
- gymnasium == 0.28.1
- pybullet
- matplotlib
- tensorboard
- gym-pybullet-drones @ git+https://github.com/utiasDSL/gym-pybullet-drones.git@v1.0.0

## Training Parameters

- Algorithm: PPO (Proximal Policy Optimization)
- Learning Rate: 3e-4
- Batch Size: 64
- Number of Steps per Update: 1024
- Total Training Steps: 50,000 (configurable)

## Results

After training, the model should be able to:
- Maintain stable hover at the target height
- Recover from small disturbances
- Achieve low height error (< 0.1m on average)
- Generate smooth control actions

Training progress and evaluation metrics are saved in the `logs/` directory as plots and text output.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source. Please check the license file for details.
