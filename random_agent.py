
import numpy as np
import spacy
import re
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import pandas as pd
import matplotlib.pyplot as plt
from Control_RI_simultanous_DQN_explorer import TextWorldEnv
import Pretraining

import os
from z8file_to_dictionaries import z8file_to_dictionaries

def run_random_agent(env, n_episodes=100):
    episode_rewards = []

    for episode in range(n_episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Extract admissible actions from the observation
            admissible_actions = np.where(observation[:8] > 0)[0]
            if len(admissible_actions) == 0:
                action = env.action_space.sample()
            else:
                # Randomly select an action from admissible actions
                action = np.random.choice(admissible_actions)

            # Take a step in the environment
            observation, reward, terminate, truncated, _ = env.step(action)
            total_reward += reward

            done = terminate or truncated

        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward, episode_rewards
def evaluate_random_agent_on_all_envs():
    # Initialize list to collect evaluation results
    results = []

    # Iterate through each subfolder in 'data/trained'
    for subfolder in os.listdir('data/trained'):
        model_path = f'data/trained/{subfolder}/Models/final_modeldict.zip'

        # Check if model file exists to ensure we have a trained model to compare
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping.")
            continue

        # Load game dictionary and room positions
        gameaddress = f'data/trained/{subfolder}/{subfolder}'
        try:
            game_dict, room_positions = z8file_to_dictionaries(gameaddress)
        except Exception as e:
            print(f"Failed to load game dictionary from {gameaddress}: {e}")
            continue

        # Extract x_destination and y_destination from the environment name
        parts = subfolder.split('_')
        if len(parts) >= 3:
            try:
                x_destination = float(parts[-2])
                y_destination = float(parts[-1].split('.')[0])
            except ValueError:
                print(f"Failed to parse coordinates from filename: {subfolder}")
                x_destination = 0.0
                y_destination = 0.0
        else:
            print(f"Unexpected subfolder name format: {subfolder}")
            x_destination = 0.0
            y_destination = 0.0

        # Create the TextWorldEnv with game_dict and room_positions
        env = TextWorldEnv(
            game_dict=game_dict,
            room_positions=room_positions,
            x_destination=x_destination,
            y_destination=y_destination,
            n_instructions=1  # Adjust if needed
        )

        # Evaluate the random agent
        try:
            mean_reward, std_reward, episode_rewards = run_random_agent(env, n_episodes=100)
        except Exception as e:
            print(f"Failed to evaluate random agent on {subfolder}: {e}")
            continue

        print(f"Environment: {subfolder} | Random Agent Mean reward: {mean_reward}, Std reward: {std_reward}")

        # Determine complexity based on subfolder name
        complexity = 0
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining25):
            complexity = 0.25
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining50):
            complexity = 0.5
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining75):
            complexity = 0.75
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining100):
            complexity = 1
        if subfolder.startswith('simplest'):
            complexity = 0

        # Append the results to the list
        results.append({
            'Environment': subfolder,
            'Random Agent Mean Reward': mean_reward,
            'Random Agent Std Reward': std_reward,
            'Complexity_of_Environment': complexity
        })

        print(f"Random agent evaluated on {subfolder}")

    # Create DataFrame from results
    df_random = pd.DataFrame(results)

    # Save the DataFrame to CSV
    df_random.to_csv('data/evaluation_result_random_agent.csv', index=False)
    print(df_random)

    print("Random agent evaluated on all environments and results saved to 'data/evaluation_result_random_agent.csv'")

    return df_random
def evaluate_all_trained_models():
    # Initialize list to collect evaluation results
    results = []

    # Iterate through each subfolder in 'data/trained'
    for subfolder in os.listdir('data/trained'):
        model_path = f'data/trained/{subfolder}/Models/final_modeldict.zip'

        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping.")
            continue

        # Load the model
        try:
            model = DQN.load(model_path)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            continue

        # Load game dictionary and room positions
        gameaddress = f'data/trained/{subfolder}/{subfolder}'
        try:
            game_dict, room_positions = z8file_to_dictionaries(gameaddress)
        except Exception as e:
            print(f"Failed to load game dictionary from {gameaddress}: {e}")
            continue

        # Extract x_destination and y_destination from the environment name
        parts = subfolder.split('_')
        if len(parts) >= 3:
            try:
                x_destination = float(parts[-2])
                y_destination = float(parts[-1].split('.')[0])
            except ValueError:
                print(f"Failed to parse coordinates from filename: {subfolder}")
                x_destination = 0.0
                y_destination = 0.0
        else:
            print(f"Unexpected subfolder name format: {subfolder}")
            x_destination = 0.0
            y_destination = 0.0

        # Create the TextWorldEnv with game_dict and room_positions
        env = TextWorldEnv(
            game_dict=game_dict,
            room_positions=room_positions,
            x_destination=x_destination,
            y_destination=y_destination,
            n_instructions=1  # Adjust if needed
        )

        # Evaluate the trained model
        try:
            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=100,
                deterministic=False,
                render=False,
                callback=None,
                reward_threshold=None,
                return_episode_rewards=False,
                warn=False
            )
        except Exception as e:
            print(f"Failed to evaluate model {subfolder}: {e}")
            continue

        print(f"Environment: {subfolder} | Trained Agent Mean reward: {mean_reward}, Std reward: {std_reward}")

        # Determine complexity based on subfolder name
        complexity = 0
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining25):
            complexity = 0.25
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining50):
            complexity = 0.5
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining75):
            complexity = 0.75
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining100):
            complexity = 1
        if subfolder.startswith('simplest'):
            complexity = 0

        # Append the results to the list
        results.append({
            'Environment': subfolder,
            'Trained Agent Mean Reward': mean_reward,
            'Trained Agent Std Reward': std_reward,
            'Complexity_of_Environment': complexity
        })

        print(f"Trained model {subfolder} evaluated successfully")

    # Create DataFrame from results
    df_trained = pd.DataFrame(results)

    # Save the DataFrame to CSV
    df_trained.to_csv('data/evaluation_result_trained_agents.csv', index=False)
    print(df_trained)

    print("All trained models evaluated and results saved to 'data/evaluation_result_trained_agents.csv'")

    return df_trained
def compare_agents():
    df_trained = evaluate_all_trained_models()
    df_random = evaluate_random_agent_on_all_envs()

    # Merge DataFrames on Environment and Complexity
    df_comparison = pd.merge(df_trained, df_random, on=['Environment', 'Complexity_of_Environment'])

    # Calculate performance difference
    df_comparison['Performance Improvement'] = df_comparison['Trained Agent Mean Reward'] - df_comparison['Random Agent Mean Reward']

    # Save the comparison to CSV
    df_comparison.to_csv('data/agents_comparison.csv', index=False)
    print(df_comparison)

    print("Agents compared and results saved to 'data/agents_comparison.csv'")

    return df_comparison

def plot_comparison(df_comparison):
    complexities = df_comparison['Complexity_of_Environment'].unique()
    complexities.sort()

    trained_means = []
    random_means = []

    for complexity in complexities:
        df_subset = df_comparison[df_comparison['Complexity_of_Environment'] == complexity]
        trained_means.append(df_subset['Trained Agent Mean Reward'].mean())
        random_means.append(df_subset['Random Agent Mean Reward'].mean())

    plt.figure(figsize=(10, 6))
    plt.plot(complexities, trained_means, label='Trained Agent')
    plt.plot(complexities, random_means, label='Random Agent')
    plt.xlabel('Environment Complexity')
    plt.ylabel('Mean Reward')
    plt.title('Agent Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    print(torch.cuda.is_available())

    # Evaluate and compare agents
    df_comparison = compare_agents()
    plot_comparison(df_comparison)

