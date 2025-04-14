import random
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from z8file_to_dictionaries import z8file_to_dictionaries
from environment import TextWorldEnv, get_admissible_actions_from_observation, extract_distance_from_observation
from utils import determine_complexity, assign_complexity_category_updated, load_envs, get_all_possible_envs

def evaluate_random_agent(env, n_eval_episodes=100, verbose=False):
    successes = []
    for episode in range(n_eval_episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        episode_actions = []
        episode_distances = []
        reward = 0

        while not done and not truncated:
            admissible_actions = get_admissible_actions_from_observation(observation, env.directions)
            if not admissible_actions:
                action = env.max_directions
            else:
                action = random.choice(admissible_actions)
            episode_actions.append(action)
            observation, reward, done, truncated, _ = env.step(action)
            distance = extract_distance_from_observation(observation)
            episode_distances.append(distance)

        successes.append(reward)
        if verbose:
            print(f"Episode {episode + 1}:")
            print(f"  Actions Taken: {episode_actions}")
            print(f"  Distances: {episode_distances}")
            print(f"  Success: {reward}")
    return successes

def get_admissible_actions_from_observation(observation, directions):
    admissible_flags = observation[:len(directions)]
    admissible_actions = [action for action, flag in enumerate(admissible_flags) if flag == 1]
    return admissible_actions

def extract_distance_from_observation(observation):
    from environment import MAX_DISTANCE
    distance = observation[-2] * MAX_DISTANCE
    return distance

def evaluate_curriculum_models(max_seen_envs=5, max_unseen_envs=5, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    grammars = [8, 6, 4]
    reward_types = ['sparse', 'step_cost']
    results = []
    n_instructions = 10

    all_env_pretraining = load_envs()
    seen_env_names = {env['env'] for env in all_env_pretraining}
    all_envs = get_all_possible_envs()
    unseen_envs = [env for env in all_envs if env['env'] not in seen_env_names]

    selected_seen = random.sample(all_env_pretraining, min(max_seen_envs, len(all_env_pretraining)))
    selected_unseen = random.sample(unseen_envs, min(max_unseen_envs, len(unseen_envs)))
    eval_envs = selected_seen + selected_unseen

    for reward_type in reward_types:
        print(f"\n=== Evaluating {reward_type} reward model ===")
        model_path = f'data/trained/curriculum/PPO/{reward_type}/best_model.zip'
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Skipping.")
            continue

        try:
            model = PPO.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Loaded {reward_type} model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        for grammar in grammars:
            print(f"\nEvaluating grammar {grammar}")
            for env_info in tqdm(eval_envs, desc="Environments"):
                env_name = env_info['env']
                is_seen = 'seen' if env_name in seen_env_names else 'unseen'
                try:
                    game_dict, room_positions = z8file_to_dictionaries(f'data/Environments/{env_name}')
                except Exception as e:
                    print(f"Skipping {env_name}: {e}")
                    continue

                env = TextWorldEnv(
                    game_dict=game_dict,
                    room_positions=room_positions,
                    n_instructions=n_instructions,
                    grammar=grammar,
                    reward_type=reward_type
                )
                env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

                success_trained, std_trained = evaluate_model(model, env)
                random_rewards = evaluate_random_agent(env, n_eval_episodes=100)
                random_successes = [1 if r >= 25 else 0 for r in random_rewards]
                success_random = np.mean(random_successes)
                std_random = np.std(random_successes)

                results.append(create_result_entry(
                    model_name=f"curriculum_{reward_type}",
                    env_name=env_name,
                    complexity=determine_complexity(env_name),
                    grammar=grammar,
                    instruction_type='complete',
                    reward_type=reward_type,
                    average_success_rate=abs(success_trained)/100,
                    std_success_rate=std_trained,
                    random_agent_average_success_rate=success_random,
                    random_agent_std_success_rate=std_random,
                    is_seen=is_seen
                ))

                if len(env.route_instructions) >= 3:
                    env_incomplete = create_incomplete_environment(env, game_dict, room_positions, grammar, reward_type, n_instructions=n_instructions)
                    env_incomplete = gym.wrappers.TimeLimit(env_incomplete, max_episode_steps=n_instructions + 14)

                    success_trained_inc, std_trained_inc = evaluate_model(model, env_incomplete)
                    random_rewards_inc = evaluate_random_agent(env_incomplete, n_eval_episodes=100)
                    random_successes_inc = [1 if r >= 25 else 0 for r in random_rewards_inc]
                    success_random_inc = np.mean(random_successes_inc)
                    std_random_inc = np.std(random_successes_inc)

                    results.append(create_result_entry(
                        model_name=f"curriculum_{reward_type}",
                        env_name=env_name,
                        complexity=determine_complexity(env_name),
                        grammar=grammar,
                        instruction_type='incomplete',
                        reward_type=reward_type,
                        average_success_rate=abs(success_trained_inc)/100,
                        std_success_rate=std_trained_inc,
                        random_agent_average_success_rate=abs(success_random_inc),
                        random_agent_std_success_rate=std_random_inc,
                        is_seen=is_seen
                    ))

    df = pd.DataFrame(results)
    df.to_csv('data/curriculum_evaluation_results.csv', index=False)
    print("\nEvaluation complete. Results saved to curriculum_evaluation_results.csv")
    return df

def evaluate_model(model, env, n_episodes=10):
    try:
        evaluation_result = evaluate_policy(
            model.policy, env, n_eval_episodes=n_episodes,
            deterministic=True, warn=False, return_episode_rewards=False
        )
        if not isinstance(evaluation_result, tuple) or len(evaluation_result) != 2:
            mean_success = np.nan
            std_success = np.nan
        else:
            rewards, _ = evaluation_result
            if not isinstance(rewards, (int, float, np.number)):
                print(f"!!! WARNING: Unexpected type for 'rewards' (mean reward): {type(rewards)}, expected number")
                mean_success = np.nan
                std_success = np.nan
            else:
                mean_success = rewards
                std_success = evaluation_result[1]
        return mean_success, std_success
    except Exception as e:
        print(f"!!! Exception in evaluate_model !!!: {e}")
        print(f"Returning NaN, NaN from except block...")
        return np.nan, np.nan

def create_incomplete_environment(base_env, game_dict, room_positions, grammar, reward_type, n_instructions=4):
    if len(base_env.route_instructions) < 3:
        return None
    incomplete_instructions = base_env.route_instructions.copy()
    del incomplete_instructions[len(incomplete_instructions) // 2]
    env = TextWorldEnv(
        game_dict=game_dict,
        room_positions=room_positions,
        n_instructions=n_instructions,
        grammar=grammar,
        reward_type=reward_type,
        is_incomplete=True,
        x_destination=base_env.x_destination,
        y_destination=base_env.y_destination,
        route_instructions=incomplete_instructions)
    return env

def create_result_entry(**kwargs):
    return {
        'model': kwargs['model_name'],
        'environment': kwargs['env_name'],
        'complexity': assign_complexity_category_updated(kwargs['complexity']),
        'grammar': kwargs['grammar'],
        'instruction_type': kwargs['instruction_type'],
        'reward_type': kwargs['reward_type'],
        'average_success_rate': round(kwargs['average_success_rate'], 3),
        'std_success_rate': round(kwargs['std_success_rate'], 3),
        'random_agent_average_success_rate': round(kwargs['random_agent_average_success_rate'], 3),
        'random_agent_std_success_rate': round(kwargs['random_agent_std_success_rate'], 3),
        'evaluated_env': kwargs['is_seen']
    }
