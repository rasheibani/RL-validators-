# Multi-Origin Generalization for Route Instruction Following

## Overview

This extension enhances the reinforcement learning agent's generalization capabilities by training it with multiple origins for each destination. The goal is to improve the agent's ability to navigate from completely new starting positions to familiar destinations.

## Features

- **Multi-Origin Training**: Trains the agent using multiple sets of route instructions for each destination, with each set starting from a different origin room
- **Novel Origin Generalization**: Evaluates the agent's ability to navigate from previously unseen origins to familiar destinations
- **Comparative Analysis**: Compares performance against baseline agents trained with single origins
- **Detailed Visualizations**: Provides charts for success rates, path efficiency, and generalization improvements

## Implementation Details

The multi-origin approach includes several key components:

1. **MultiOriginTrainingData**: Class that generates training data with multiple origins for each destination
   - Finds reachable origins for destinations using BFS
   - Generates route instructions for each origin-destination pair
   - Reserves some origins for evaluation

2. **MultiOriginEnv**: Extended environment that uses multiple origins during training
   - Randomly selects origin-destination pairs during training
   - Uses route instructions specific to each origin-destination pair

3. **Evaluation**: Compares multi-origin trained agents against baseline agents
   - Measures success rates and path efficiency
   - Tests on previously unseen origins to evaluate generalization

## Usage

### Running the Multi-Origin Experiment

```bash
python main_multi_origin.py --num_origins 5 --num_envs 5 --max_iterations 50000
```

Parameters:
- `--num_origins`: Number of different origins to use per destination (default: 5)
- `--num_envs`: Number of environments to include in the experiment (default: 5)
- `--max_iterations`: Maximum training iterations per environment (default: 50000)

### Visualizing Results

After running the experiment, you can generate visualizations:

```bash
python visualize_multi_origin.py
```

This will create charts comparing multi-origin and baseline approaches in the `charts/multi_origin/` directory.

## Expected Outcomes

The multi-origin approach should demonstrate:

1. **Improved Generalization**: Higher success rates when navigating from new origins
2. **Better Path Efficiency**: More optimal paths from origins to destinations
3. **Robust Navigation**: Less dependence on specific paths learned during training

## Files

- `multi_origin_training.py`: Core implementation of multi-origin training approach
- `main_multi_origin.py`: Entry point for running the multi-origin experiment
- `run_multi_origin_experiment.py`: Script to run experiments comparing approaches
- `visualize_multi_origin.py`: Generates charts and visualizations of results

## Results

Results from the experiments will be saved to:
- `data/multi_origin_results_all.csv`: Complete comparison metrics
- `charts/multi_origin/`: Visualization charts comparing approaches

## How It Works

1. For each destination, we identify multiple origin rooms that can reach it
2. We generate route instructions from each origin to the destination
3. During training, we randomly select origin-destination pairs
4. For evaluation, we test on origins not seen during training
5. We compare against a baseline agent trained with only one origin per destination

This approach helps the agent learn more general navigation strategies rather than memorizing specific paths.