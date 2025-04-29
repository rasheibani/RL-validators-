# Reinforcement Learning for Route Instruction Following

A reinforcement learning framework for training agents to navigate environments by following natural language route instructions. This project implements curriculum learning strategies, different reward mechanisms, and a novel multi-origin training approach to improve generalization capabilities.

## Overview

This project trains reinforcement learning agents to navigate through grid-based environments by following route instructions. It focuses on generalizing navigation skills across different environments with varying complexity levels. The core research questions addressed include:

1. How effective are different reward strategies (sparse vs. step-cost) for navigation tasks?
2. Can curriculum learning improve performance on complex environments?
3. How well do agents generalize to unseen environments?
4. Can multi-origin training improve generalization to novel starting positions?

## Key Features

- **Environment Generation**: Supports various grid-based environments with customizable complexity
- **Curriculum Learning**: Progressive training from simple to complex environments
- **Multiple Reward Types**: Supports sparse rewards and step-cost rewards
- **Natural Language Processing**: Processes route instructions into directional actions
- **Performance Evaluation**: Comprehensive metrics for measuring navigation success
- **Multi-Origin Training**: Novel approach for improving generalization from unseen origins

## Project Structure

- **Main Components**:
  - `environment.py`: Implements the TextWorldEnv reinforcement learning environment
  - `training.py`: Contains the curriculum learning implementation
  - `utils.py`: Utility functions for environment handling and complexity metrics
  - `evaluation.py`: Functions for evaluating trained models
  - `main.py`: Entry point for standard curriculum training approach
  - `multi_origin_training.py`: Implementation of multi-origin training approach
  - `main_multi_origin.py`: Entry point for multi-origin experiments
  - `visualize_multi_origin.py`: Visualization tools for multi-origin results

- **Data Structure**:
  - `data/Environments/`: Contains environment files (.z8 format)
  - `data/RouteInstructions/`: Route instruction datasets
  - `data/trained/`: Saved model checkpoints
  - `charts/`: Visualizations of training and evaluation results

## Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Spacy (with 'en_core_web_sm' model)
- TextWorld (for environment generation)

## Usage Instructions

### Standard Curriculum Learning Approach

To train agents using the standard curriculum learning approach:

```bash
python main.py
```

This will:
1. Load environments from `data/Environments/`
2. Train models progressively on environments sorted by complexity
3. Save trained models to `data/trained/curriculum/PPO/`
4. Evaluate models on both seen and unseen environments

### Multi-Origin Training Approach

To train agents using the multi-origin approach (for improved generalization from novel starting positions):

```bash
python main_multi_origin.py --num_origins 5 --num_envs 5 --max_iterations 50000
```

Parameters:
- `--num_origins`: Number of different origins to use per destination (default: 5)
- `--num_envs`: Number of environments to include in the experiment (default: 5)
- `--max_iterations`: Maximum training iterations per environment (default: 50000)

### Visualizing Results

To visualize the results of multi-origin experiments:

```bash
python visualize_multi_origin.py
```

This generates charts in the `charts/multi_origin/` directory comparing multi-origin and baseline approaches.

## Multi-Origin Training Extension

### Concept

The multi-origin training extension enhances the agent's ability to navigate from completely new starting positions to familiar destinations. Traditional navigation models often struggle with generalization when starting from unfamiliar origins. This extension addresses this limitation by:

1. Training on multiple sets of route instructions for each destination
2. Using different origin rooms for each instruction set
3. Evaluating on completely new origins not encountered during training

### Implementation Details

- **Data Generation**: For each destination, identifies multiple origin rooms and generates route instructions
- **Training Strategy**: Randomly selects origin-destination pairs during training
- **Evaluation**: Compares against baseline agents trained with single origins
- **Metrics**: Measures success rates and path efficiency improvements

### Expected Benefits

1. **Robust Navigation**: Reduced dependence on memorized paths
2. **Better Generalization**: Improved ability to navigate from unseen starting positions
3. **Efficient Paths**: More optimal navigation from origins to destinations

## Reinforcement Learning Configuration

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Policy Network**: MLP with [64, 64] hidden layers
- **Learning Rate**: 0.001
- **Reward Types**:
  - Sparse: Rewards only at successful completion
  - Step-cost: Penalizes each step (-1) with large reward at completion (+25)
- **Episode Termination**: Upon reaching destination or exceeding maximum steps

## Evaluation Metrics

- **Success Rate**: Percentage of episodes with successful navigation
- **Path Efficiency**: Ratio of optimal path length to actual path length
- **Generalization Gap**: Performance difference between seen and unseen environments
- **Learning Efficiency**: Success rate improvement over training iterations

## Results Analysis

The project includes visualizations for analyzing agent performance:

- Success rates across different environment complexities
- Comparison between reward types
- Path efficiency metrics
- Generalization capabilities to unseen environments
- Multi-origin vs. baseline performance comparisons

## Extending the Project

To extend this project with new features:

1. **New Environments**: Add .z8 files to `data/Environments/`
2. **Custom Reward Functions**: Modify the reward calculation in `environment.py`
3. **Alternative RL Algorithms**: Create new training implementations based on the existing PPO approach
4. **Enhanced Instructions**: Update route instruction processing in `environment.py`

## References

- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- TextWorld: https://github.com/microsoft/TextWorld
- Gymnasium: https://gymnasium.farama.org/