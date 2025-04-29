import random
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
import re

from pydantic.v1.utils import truncate
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import os
from z8file_to_dictionaries import z8file_to_dictionaries
from tqdm import tqdm  # For progress bars

import Pretraining

# Define grammars and their corresponding directions
GRAMMAR_DIRECTIONS = {
    4: ['north', 'south', 'east', 'west'],
    6: ['north', 'south', 'east', 'west', 'northeast', 'southwest'],  # Example for 6 sectors
    8: ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']
}

# Maximum distance for normalization (adjust based on your environment)
MAX_DISTANCE = 2000.0  # Example value

def extract_coordinates(game_state):
    # Regular expression pattern to match the X and Y values
    pattern = r"X:\s*([\d.]+)\s*\nY:\s*([\d.]+)"

    matches = re.search(pattern, game_state)
    if matches:
        x = float(matches.group(1))
        y = float(matches.group(2))
        return np.array([x, y])
    else:
        return np.array([0, 0])

def text_to_action(text, directions):
    """
    Converts a text instruction to an action index based on the available directions.

    Parameters:
    - text (str): The action in text form (e.g., "go north").
    - directions (list of str): The list of directions corresponding to the current grammar.

    Returns:
    - int: The action index or -1 if not found.
    """
    mapping = {direction: idx for idx, direction in enumerate(directions)}
    return mapping.get(text.strip().lower(), -1)

def sentence_from_action(action, directions):
    """
    Converts an action index to a sentence based on the provided directions.

    Parameters:
    - action (int): The action index.
    - directions (list of str): The list of directions corresponding to the current grammar.

    Returns:
    - str: The sentence representing the action.
    """
    if 0 <= action < len(directions):
        return f"go {directions[action]}"
    else:
        return "look"

def normalize(observation):
    """
    Normalize the observation to be in the correct range and shape.
    """
    # Separate components (assuming shape is max_directions + 15 + 2)
    max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
    admissible_actions = observation[:max_directions]
    route_instructions = observation[max_directions:-2]
    instruction_indices = observation[-2:]  # Both index and current instruction

    # Normalize admissible actions (already in [0, 1])
    normalized_admissible_actions = admissible_actions

    # Normalize route instructions
    normalized_route_instructions = np.where(
        route_instructions != max_directions,
        route_instructions / (max_directions - 1),
        -1
    )

    # Normalize instruction indices
    max_instruction_index = len(route_instructions)
    normalized_indices = np.array([
        instruction_indices[0] / max(max_instruction_index, 1),  # instruction index
        instruction_indices[1] / (max_directions - 1) if instruction_indices[1] != max_directions else -1  # current instruction
    ])

    # Combine all components
    normalized_observation = np.concatenate([
        normalized_admissible_actions,
        normalized_route_instructions,
        normalized_indices
    ])

    return normalized_observation
def get_admissible_actions(feedback, directions):
    admissible_actions = []
    for direction in directions:
        pattern = r'\bgoing ' + direction + r'\b'  # Added word boundaries to prevent partial matches
        if re.search(pattern, feedback, re.IGNORECASE):
            admissible_actions.append('go ' + direction)
    return admissible_actions


def admissible_actions_to_observation(admissible_actions, directions):
    """
    Convert admissible actions to binary vector with consistent shape.
    """
    max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
    observation = np.zeros(max_directions, dtype=np.int32)

    for i, direction in enumerate(directions):
        if f"go {direction}" in admissible_actions:
            observation[i] = 1

    return observation
def extract_area_id(feedback):
    pattern = r"An area \((\d+)\) in r(\d+)"
    matches = re.search(pattern, feedback)
    if matches:
        area_id = matches.group(1)
        room_id = matches.group(2)
        return f"a{area_id}r{room_id}"
    else:
        # Debugging: Print feedback when extraction fails
        print("Failed to extract area ID from feedback:")
        print(feedback)
        print("-" * 50)
        return None

class TextWorldEnv(gym.Env):
    def __init__(self, game_dict, room_positions, x_destination=None, y_destination=None, n_instructions=5, grammar=8, reward_type='sparse'):
        super(TextWorldEnv, self).__init__()
        self.game_dict = game_dict  # The game dictionary
        self.room_positions = room_positions  # Mapping from room IDs to (x, y) coordinates
        self.current_room_id = None
        self.n_instructions = n_instructions
        self.grammar = grammar  # 4, 6, or 8
        self.reward_type = reward_type  # 'sparse' or 'step_cost'
        self.instruction_index = 0
        self.route_instructions = []
        self.visited_states_actions = set()
        self.last_feedback_embedding = None
        self.counter = 0
        self.x_destination = x_destination
        self.y_destination = y_destination
        self.x_origin = None
        self.y_origin = None
        self.exploration_threshold = 0

        # Define actions based on grammar
        if self.grammar in GRAMMAR_DIRECTIONS:
            self.directions = GRAMMAR_DIRECTIONS[self.grammar]
        else:
            raise ValueError("Invalid grammar. Choose from 4, 6, or 8.")

        self.max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
        self.action_space = spaces.Discrete(self.max_directions + 1)  # +1 for "look"
        self.observation_space = spaces.Box(low=0, high=8, shape=(self.max_directions + 15 + 2,), dtype=np.int32)

    def text_to_action_func(self, text):
        return text_to_action(text, self.directions)

    def sentence_from_action_func(self, action):
        return sentence_from_action(action, self.directions)

    def generate_route_instructions(self):
        route_instructions = []
        temp_room_id = self.current_room_id
        visited_rooms = set([temp_room_id])

        for _ in range(self.n_instructions):
            admissible_actions = list(self.game_dict[temp_room_id].keys())
            if not admissible_actions:
                break

            # Filter out actions leading to already visited rooms to prevent loops
            admissible_actions = [
                action for action in admissible_actions
                if self.game_dict[temp_room_id][action] not in visited_rooms
            ]

            if not admissible_actions:
                break

            action = np.random.choice(admissible_actions)
            route_instructions.append(self.text_to_action_func(action))
            temp_room_id = self.game_dict[temp_room_id][action]
            visited_rooms.add(temp_room_id)

            if temp_room_id is None:
                break

        self.x_destination, self.y_destination = self.room_positions[temp_room_id]
        return route_instructions, self.x_destination, self.y_destination

    def get_destination_from_route_instructions(self, route_instructions):
        temp_room_id = self.current_room_id
        for action in route_instructions:
            action_text = self.sentence_from_action_func(action)
            next_room_id = self.game_dict[temp_room_id].get(action_text)
            if next_room_id is None:
                break  # Invalid action
            temp_room_id = next_room_id
        return self.room_positions[temp_room_id]

    def reset(self, **kwargs):
        self.counter = 0
        # Set the current room to a random starting room
        self.current_room_id = random.choice(list(self.game_dict.keys()))
        self.x, self.y = self.room_positions[self.current_room_id]
        self.x_origin, self.y_origin = self.x, self.y
        self.visited_states_actions.clear()
        self.instruction_index = 0

        if 'route_instructions' in kwargs:
            rti = kwargs['route_instructions']
            # Handle incomplete route instructions by allowing missing steps
            self.route_instructions = [self.text_to_action_func(instr) for instr in rti.split('. ')]
            self.x_destination, self.y_destination = self.get_destination_from_route_instructions(self.route_instructions)
        else:
            self.route_instructions, self.x_destination, self.y_destination = self.generate_route_instructions()

        self.dist_from_origin_to_destination = np.sqrt(
            (self.x_destination - self.x_origin) ** 2 + (self.y_destination - self.y_origin) ** 2
        )

        admissible_actions = self.get_admissible_actions()
        observation = np.concatenate((
            admissible_actions_to_observation(admissible_actions, self.directions),
            self.pad_instructions(),
            np.array([self.instruction_index, self.route_instructions[self.instruction_index]]),
        ))
        # verify shape
        assert observation.shape[0] == self.observation_space.shape[0]\
            , f"Observation shape mismatch. Expected {self.observation_space.shape[0]}, got {observation.shape[0]}"

        observation = normalize(observation)
        return observation, {}

    def get_admissible_actions(self):
        return list(self.game_dict[self.current_room_id].keys())

    def pad_instructions(self):
        if len(self.route_instructions) < 15:
            return np.pad(
                self.route_instructions,
                (0, 15 - len(self.route_instructions)),
                'constant',
                constant_values=self.max_directions  # Use max_directions as padding value
            )
        else:
            return self.route_instructions[:15]

    def construct_observation(self, admissible_actions):
        # Get current instruction safely
        current_instruction = (
            self.route_instructions[self.instruction_index]
            if self.instruction_index < len(self.route_instructions)
            else self.max_directions
        )

        # Build observation with consistent shapes
        observation = np.concatenate([
            admissible_actions_to_observation(admissible_actions, self.directions),  # max_directions elements
            self.pad_instructions(),  # 15 elements
            np.array([self.instruction_index, current_instruction])  # 2 elements
        ])

        # Verify shape before normalization
        expected_shape = self.observation_space.shape[0]
        assert observation.shape[
                   0] == expected_shape, f"Observation shape mismatch. Expected {expected_shape}, got {observation.shape[0]}"

        return normalize(observation)
    def step(self, action):
        sentence = self.sentence_from_action_func(action)
        admissible_actions = self.get_admissible_actions()
        terminate = False
        truncated = False
        self.counter += 1


        # Define "look" action
        look_action_index = self.max_directions
        if action == look_action_index:
            if self.reward_type == 'sparse':
                reward = -1
            elif self.reward_type == 'step_cost':
                reward = -1  # Step cost
            terminate = False
            truncated = False
            observation = self.construct_observation(admissible_actions)
            return observation, reward, terminate, truncated, {}

        if sentence not in admissible_actions:
            if self.reward_type == 'sparse':
                reward = -1
            elif self.reward_type == 'step_cost':
                reward = -1  # Step cost for invalid action
            terminate = False
            truncated = False
            observation = self.construct_observation(admissible_actions)
            return observation, reward, terminate, truncated, {}
        else:
            # Move to the next room
            next_room_id = self.game_dict[self.current_room_id][sentence]
            if next_room_id is None:
                if self.reward_type == 'sparse':
                    reward = -1
                elif self.reward_type == 'step_cost':
                    reward = -1  # Step cost for invalid action
                terminate = False
                truncated = False
                observation = self.construct_observation(admissible_actions)
                return observation, reward, terminate, truncated, {}

            self.current_room_id = next_room_id
            self.x, self.y = self.room_positions[self.current_room_id]

            # Calculate reward
            target_x = self.x_destination
            target_y = self.y_destination

            if np.isclose(self.x, target_x, atol =1e-3) and np.isclose(self.y, target_y, atol=1e-3):
                reward = 25
                terminate = True
                truncated = False
                observation = self.construct_observation(admissible_actions)
                # print("terminate1", terminate)
            else:
                if self.reward_type == 'step_cost':
                    reward = -0.5  # Step cost
                    # stop long exploration (after 30 steps)
                    if self.counter > self.n_instructions + self.exploration_threshold:
                        reward = -1
                        terminate = False
                        truncated = True
                        observation = self.construct_observation(admissible_actions)
                        return observation, reward, terminate, truncated, {}
                elif self.reward_type == 'sparse':
                    # Check for step limit
                    if self.instruction_index >= len(self.route_instructions) + self.exploration_threshold - 1:
                        reward = -1
                        terminate = False
                        truncated = True
                        observation = self.construct_observation(admissible_actions)
                        self.instruction_index += 1
                        return observation, reward, terminate, truncated, {}
                    reward = 0
            # print("terminate2", terminate) if terminate else None
                # terminate = False
                # truncated = False

            admissible_actions = self.get_admissible_actions()
            observation = self.construct_observation(admissible_actions)
            self.instruction_index += 1
            return observation, reward, terminate, truncated, {}



    def render(self):
        pass  # Implement if needed

    def close(self):
        pass  # Implement if needed

    def __len__(self):
        return 1  # Only one game running at a time

def determine_complexity(env_name):
    """
    Determines the complexity level based on the environment name.

    Parameters:
    - env_name (str): The name of the environment file.

    Returns:
    - float: The complexity level.
    """
    complexity = 0.0
    if any(env_name.startswith(envP) for envP in Pretraining.Pretraining25):
        complexity = 0.25
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining50):
        complexity = 0.5
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining75):
        complexity = 0.75
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining100):
        complexity = 1.0
    elif env_name.startswith('simplest'):
        complexity = 0.0
    else:
        complexity = 0.0  # Default value
    return complexity

def assign_complexity_category_updated(complexity):
    """
    Assigns a complexity category based on the complexity value.

    Parameters:
    - complexity (float): The complexity level.

    Returns:
    - str: The complexity category.
    """
    if complexity < 0.26:
        return '0-0.26'
    elif 0.26 <= complexity < 0.51:
        return '0.26-0.51'
    elif 0.51 <= complexity < 0.76:
        return '0.51-0.76'
    elif 0.76 <= complexity <= 1:
        return '0.76-1'
    else:
        return 'Unknown'

def load_envs():
    """
    Loads all environments used during pretraining.

    Returns:
    - list of dict: Each dict contains environment details.
    """
    pretraining_set = Pretraining.Pretraining25 + Pretraining.Pretraining50 + \
                      Pretraining.Pretraining75 + Pretraining.Pretraining100
    all_env_pretraining = []
    for env_prefix in pretraining_set:
        for file in os.listdir('data/Environments'):
            if file.startswith(env_prefix) and file.endswith('.z8'):
                env_name = file
                env_name_parts = env_name.split('_')
                try:
                    x_destination = float(env_name_parts[-2])
                    y_destination = float(env_name_parts[-1].split('.')[0])
                except (IndexError, ValueError):
                    x_destination = 0.0
                    y_destination = 0.0
                all_env_pretraining.append({
                    'env': file,
                    'x_destination': x_destination,
                    'y_destination': y_destination
                })
    return all_env_pretraining

def get_all_possible_envs():
    """
    Retrieves all possible environments from the 'data/Environments' directory.

    Returns:
    - list of dict: Each dict contains environment details.
    """
    all_envs = []
    for file in os.listdir('data/Environments'):
        if file.endswith('.z8'):
            env_name = file
            env_name_parts = env_name.split('_')
            try:
                x_destination = float(env_name_parts[-2])
                y_destination = float(env_name_parts[-1].split('.')[0])
            except (IndexError, ValueError):
                x_destination = 0.0
                y_destination = 0.0
            n_instructions = determine_n_instructions(env_name)
            all_envs.append({
                'env': file,
                'x_destination': x_destination,
                'y_destination': y_destination,
                'n_instructions': n_instructions
            })
    return all_envs

def determine_n_instructions(env_name):
    """
    Determines the number of instructions based on the environment name.

    Parameters:
    - env_name (str): The name of the environment file.

    Returns:
    - int: The number of instructions.
    """
    if any(env_name.startswith(envP) for envP in Pretraining.Pretraining25):
        n_instructions = 2
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining50):
        n_instructions = 3
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining75):
        n_instructions = 4
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining100):
        n_instructions = 5
    elif env_name.startswith('simplest'):
        n_instructions = 1
    else:
        n_instructions = 1  # Default value
    return n_instructions

def evaluate_random_agent(env, n_eval_episodes=100, verbose=False):
    """
    Evaluates a random agent on the given environment.

    Parameters:
    - env (gym.Env): The environment to evaluate on.
    - n_eval_episodes (int): Number of episodes to run for evaluation.
    - verbose (bool): If True, prints detailed information for each episode.

    Returns:
    - average_success_rate (float): The average success rate over all episodes.
    - std_success_rate (float): The standard deviation of the success rates.
    """
    successes = []

    for episode in range(n_eval_episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        episode_actions = []
        episode_distances = []

        while not done and not truncated:
            # Extract admissible actions from the observation
            admissible_actions = get_admissible_actions_from_observation(observation, env.directions)

            if not admissible_actions:
                # No admissible actions available, take a default action ("look")
                action = env.max_directions  # Assuming "look" is the last action
            else:
                # Select a random action from admissible actions
                action = random.choice(admissible_actions)

            episode_actions.append(action)

            # Take the action in the environment
            observation, reward, done, truncated, _ = env.step(action)

            # Optionally, track distance or other metrics if available
            distance = extract_distance_from_observation(observation)
            episode_distances.append(distance)

        # Determine success based on reward
        success = 1 if reward >= 25 else 0
        successes.append(success)

        if verbose:
            print(f"Episode {episode + 1}:")
            print(f"  Actions Taken: {episode_actions}")
            print(f"  Distances: {episode_distances}")
            print(f"  Success: {success}")

    average_success_rate = sum(successes) / len(successes)
    std_success_rate = np.std(successes)

    return average_success_rate, std_success_rate

def get_admissible_actions_from_observation(observation, directions):
    """
    Extracts admissible action indices from the normalized observation.

    Parameters:
    - observation (np.array): The normalized observation from the environment.
    - directions (list of str): The list of directions corresponding to the current grammar.

    Returns:
    - list of int: Indices of admissible actions.
    """
    admissible_flags = observation[:len(directions)]
    admissible_actions = [action for action, flag in enumerate(admissible_flags) if flag == 1]
    return admissible_actions

def extract_distance_from_observation(observation):
    """
    Extracts the distance from the normalized observation.

    Parameters:
    - observation (np.array): The normalized observation from the environment.

    Returns:
    - float: The current distance to the destination.
    """
    # Assuming that 'distance' is the second last element of the observation
    # Adjust the index based on your observation structure
    distance = observation[-2] * MAX_DISTANCE  # Replace MAX_DISTANCE with your scaling factor
    return distance

def evaluate_all_trained_models(max_seen_envs_per_model=5, max_unseen_envs_per_model=5, random_seed=42):
    """
    Evaluates all trained models on both seen and unseen environments across different grammars,
    limiting the number of environments evaluated per model, and compares with a random agent's performance.

    Parameters:
    - max_seen_envs_per_model (int): Maximum number of seen environments to evaluate per model.
    - max_unseen_envs_per_model (int): Maximum number of unseen environments to evaluate per model.
    - random_seed (int): Seed for random number generator to ensure reproducibility.
    """

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Define the grammars to evaluate
    grammars = [4, 6, 8]

    # Load all environments used during training (seen environments)
    all_env_pretraining = load_envs()  # Environments used during training

    # Get all possible environments
    all_envs = get_all_possible_envs()  # Function to get all possible environments

    # Determine unseen environments
    seen_env_names = {env['env'] for env in all_env_pretraining}
    unseen_envs = [env for env in all_envs if env['env'] not in seen_env_names]

    # Initialize list to collect evaluation results
    results = []

    # Iterate through each trained model with progress bar
    trained_models = sorted(os.listdir('data/trained'))
    for model_index, model_folder in enumerate(tqdm(trained_models, desc="Evaluating Models")):
        model_dir = f'data/trained/{model_folder}/Models'
        if not os.path.isdir(model_dir):
            print(f"No Models directory found in {model_dir}. Skipping.")
            continue

        # Define reward types
        reward_types = ['sparse', 'step_cost']

        for reward_type in reward_types:
            model_path = f'data/trained/{model_folder}/Models/final_modeldict_{reward_type}.zip'

            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Model file {model_path} does not exist. Skipping.")
                continue

            # Load the model
            try:
                model = DQN.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                continue

            # Determine the environments the model was trained on (seen)
            # In curriculum learning, each model may have been trained on all previous environments
            seen_envs_for_model = all_env_pretraining[:model_index + 1]
            print(seen_envs_for_model)

            # Loop through each grammar
            for grammar in grammars:
                # Limit the number of seen environments
                if len(seen_envs_for_model) > max_seen_envs_per_model:
                    selected_seen_envs = random.sample(seen_envs_for_model, max_seen_envs_per_model)
                else:
                    selected_seen_envs = seen_envs_for_model

                # Define unseen environments for this model
                if len(unseen_envs) > max_unseen_envs_per_model:
                    selected_unseen_envs = random.sample(unseen_envs, max_unseen_envs_per_model)
                else:
                    selected_unseen_envs = unseen_envs

                # Combine selected seen and unseen environments
                eval_envs = selected_seen_envs + selected_unseen_envs

                # Iterate through each environment with progress bar
                for env_info in tqdm(eval_envs, desc=f"Grammar {grammar} - Reward {reward_type} - Model {model_folder}", leave=False):
                    env_name = env_info['env']
                    is_seen = 'seen' if env_name in seen_env_names else 'unseen'

                    # Determine the complexity of the environment
                    complexity = determine_complexity(env_name)
                    complexity = assign_complexity_category_updated(complexity)

                    # Load game dictionary and room positions
                    gameaddress = f'data/Environments/{env_name}'
                    try:
                        game_dict, room_positions = z8file_to_dictionaries(gameaddress)
                        print(f"Loaded game dictionary from {gameaddress}")
                    except Exception as e:
                        print(f"Failed to load game dictionary from {gameaddress}: {e}")
                        continue

                    # Create the TextWorldEnv with the specified grammar and reward_type
                    env = TextWorldEnv(
                        game_dict=game_dict,
                        room_positions=room_positions,
                        n_instructions=4,  # Adjust as needed
                        grammar=grammar,
                        reward_type=reward_type
                    )
                    env_logs_dir = f'data/trained/{model_folder}/Logs'
                    os.makedirs(env_logs_dir, exist_ok=True)
                    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
                    env = Monitor(env, filename=f'{env_logs_dir}/monitor.log', allow_early_resets=True)

                    # Evaluate the trained model with complete instructions (deterministic=True)
                    try:
                        episode_rewards_trained, _ = evaluate_policy(
                            model,
                            env,
                            n_eval_episodes=10,
                            deterministic=True,  # Deterministic evaluation for trained model
                            render=False,
                            callback=None,
                            reward_threshold=None,
                            return_episode_rewards=True,
                            warn=False
                        )

                        successes_trained = [1 if reward >= 20 else 0 for reward in episode_rewards_trained]
                        average_success_rate_trained = sum(successes_trained) / len(successes_trained)
                        std_success_rate_trained = np.std(successes_trained)
                    except Exception as e:
                        print(f"Failed to evaluate model {model_folder} on environment {env_name} with grammar {grammar} (Trained): {e}")
                        average_success_rate_trained = np.nan
                        std_success_rate_trained = np.nan

                    # Evaluate the trained model as a random agent (deterministic=False)
                    try:
                        episode_rewards_random, _ = evaluate_policy(
                            model,
                            env,
                            n_eval_episodes=10,
                            deterministic=False,  # Stochastic evaluation for random agent
                            render=False,
                            callback=None,
                            reward_threshold=None,
                            return_episode_rewards=True,
                            warn=False
                        )





                        successes_random = [1 if reward >= 20 else 0 for reward in episode_rewards_random]
                        average_success_rate_random = sum(successes_random) / len(successes_random)
                        std_success_rate_random = np.std(successes_random)
                    except Exception as e:
                        print(f"Failed to evaluate model {model_folder} on environment {env_name} with grammar {grammar} (Random Agent Simulation): {e}")
                        average_success_rate_random = np.nan
                        std_success_rate_random = np.nan

                    # Append the results for complete instructions
                    results.append({
                        'learned_model_name': model_folder[:10],               # New column
                        'test_env_short': env_name[:10],                  # New column
                        # 'name_of_env': env_name,
                        'complexity': complexity,
                        'grammar': grammar,
                        'instruction_type': 'complete',
                        'reward_type': reward_type,
                        'average_success_rate': round(average_success_rate_trained,3),
                        'std_success_rate': round(std_success_rate_trained,3),
                        'random_agent_average_success_rate': round(average_success_rate_random,3),
                        'random_agent_std_success_rate': round(std_success_rate_random,3),
                        'evaluated_env': is_seen
                    }) if is_seen =='unseen' or (is_seen=='seen' and model_folder==env_name) else None

                    print(f"Model {model_folder} (Reward: {reward_type}) evaluated on environment {env_name} with grammar {grammar} (Complete, {is_seen})")
                    print(f"Trained Agent - Average Success Rate: {average_success_rate_trained}, Std: {std_success_rate_trained}")
                    print(f"Random Agent Simulation - Average Success Rate: {average_success_rate_random}, Std: {std_success_rate_random}")

                    # Now evaluate with incomplete instructions by omitting one step (excluding first and last)
                    # Ensure that the route instructions have at least 3 steps to omit one
                    if len(env.route_instructions) >= 3:
                        # Choose the step to omit (e.g., the 6th step if exists, else a random step excluding first and last)
                        omit_step_index = 2
                        if len(env.route_instructions) > omit_step_index:
                            omitted_step = env.route_instructions.pop(omit_step_index)
                        else:
                            # Randomly choose a step to omit excluding first and last
                            omit_step_index = random.randint(1, len(env.route_instructions) - 2)
                            omitted_step = env.route_instructions.pop(omit_step_index)

                        # Reconstruct the incomplete route instruction string
                        incomplete_route_instruction = '. '.join([sentence_from_action(action, env.directions) for action in env.route_instructions]) + '. Arrive at destination!'

                        # Create a new environment instance for incomplete instructions
                        try:
                            env_incomplete = TextWorldEnv(
                                game_dict=game_dict,
                                room_positions=room_positions,
                                n_instructions=4,  # Must match the main env
                                grammar=grammar,
                                reward_type=reward_type
                            )
                            env_incomplete = gym.wrappers.TimeLimit(env_incomplete, max_episode_steps=100)
                            env_incomplete = Monitor(env_incomplete, filename=f'{env_logs_dir}/monitor_incomplete.log', allow_early_resets=True)

                            # Reset with incomplete instructions
                            observation_incomplete, _ = env_incomplete.reset(route_instructions=incomplete_route_instruction)

                            # Evaluate the trained model with incomplete instructions (deterministic=True)
                            try:
                                episode_rewards_trained_incomplete, _ = evaluate_policy(
                                    model,
                                    env_incomplete,
                                    n_eval_episodes=10,
                                    deterministic=True,  # Deterministic evaluation for trained model
                                    render=False,
                                    callback=None,
                                    reward_threshold=None,
                                    return_episode_rewards=True,
                                    warn=False
                                )


                                successes_trained_incomplete = [1 if reward >= 20 else 0 for reward in episode_rewards_trained_incomplete]
                                average_success_rate_trained_incomplete = sum(successes_trained_incomplete) / len(successes_trained_incomplete)
                                std_success_rate_trained_incomplete = np.std(successes_trained_incomplete)
                            except Exception as e:
                                print(f"Failed to evaluate model {model_folder} on environment {env_name} with grammar {grammar} (Incomplete, Trained): {e}")
                                average_success_rate_trained_incomplete = np.nan
                                std_success_rate_trained_incomplete = np.nan

                            # Evaluate the trained model as a random agent on incomplete instructions (deterministic=False)
                            try:
                                episode_rewards_random_incomplete, _ = evaluate_policy(
                                    model,
                                    env_incomplete,
                                    n_eval_episodes=10,
                                    deterministic=False,  # Stochastic evaluation for random agent simulation
                                    render=False,
                                    callback=None,
                                    reward_threshold=None,
                                    return_episode_rewards=True,
                                    warn=False
                                )
                                successes_random_incomplete = [1 if reward >= 20 else 0 for reward in episode_rewards_random_incomplete]
                                average_success_rate_random_incomplete = sum(successes_random_incomplete) / len(successes_random_incomplete)
                                std_success_rate_random_incomplete = np.std(successes_random_incomplete)
                            except Exception as e:
                                print(f"Failed to evaluate model {model_folder} on environment {env_name} with grammar {grammar} (Incomplete, Random Agent Simulation): {e}")
                                average_success_rate_random_incomplete = np.nan
                                std_success_rate_random_incomplete = np.nan

                            # Append the results for incomplete instructions
                            results.append({
                                'learned_model_name': model_folder[:10],               # New column
                                'test_env_short': env_name[:10],                  # New column
                                # 'name_of_env': env_name,
                                'complexity': complexity,
                                'grammar': grammar,
                                'instruction_type': 'incomplete',
                                'reward_type': reward_type,
                                'average_success_rate': round(average_success_rate_trained_incomplete,3),
                                'std_success_rate': round(std_success_rate_trained_incomplete,3),
                                'random_agent_average_success_rate': round(average_success_rate_random_incomplete,3),
                                'random_agent_std_success_rate': round(std_success_rate_random_incomplete,3),
                                'evaluated_env': is_seen
                            }) if is_seen =='unseen' or (is_seen=='seen' and model_folder==env_name) else None

                            print(f"Model {model_folder} (Reward: {reward_type}) evaluated on environment {env_name} with grammar {grammar} (Incomplete, {is_seen})")
                            print(f"Trained Agent - Average Success Rate: {average_success_rate_trained_incomplete}, Std: {std_success_rate_trained_incomplete}")
                            print(f"Random Agent Simulation - Average Success Rate: {average_success_rate_random_incomplete}, Std: {std_success_rate_random_incomplete}")

                        except Exception as e:
                            print(f"Failed to evaluate model {model_folder} on environment {env_name} with grammar {grammar} (Incomplete): {e}")
                            continue

    # After all evaluations, save the results
    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Save the DataFrame to CSV
    df.to_csv('data/evaluation_results.csv', index=False)
    print(df)

    print("All models evaluated and results saved to 'data/evaluation_results.csv'")


from stable_baselines3.common.callbacks import BaseCallback

class CustomStopOnNoImprovement(BaseCallback):
    """
    A custom callback that stops training if there is no model improvement over a given number of evaluations.

    :param max_no_improvement_evals: The number of evaluations without improvement before stopping.
    :param min_evals: The minimum number of evaluations before checking for improvement.
    :param verbose: Verbosity level (0: no output, 1: info messages, 2: debug messages)
    """

    def __init__(self, max_no_improvement_evals: int = 3, min_evals: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.eval_count = 0
        self.no_improvement_count = 0
        self.best_reward = -np.inf

    def _on_training_start(self) -> None:
        """This method is called before the first rollout starts."""
        self.eval_count = 0
        self.no_improvement_count = 0
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        """Called after each call to `env.step()`. Can stop training early."""
        return True  # We continue training by default

    def _on_rollout_end(self) -> None:
        """This method is called before updating the policy."""
        # Perform evaluation after each rollout (or batch of rollouts)
        if self.eval_count >= self.min_evals:
            eval_reward = self.evaluate_model()
            self.eval_count += 1

            # Check if the reward has improved
            if eval_reward > self.best_reward:
                self.best_reward = eval_reward
                self.no_improvement_count = 0  # Reset counter
            else:
                self.no_improvement_count += 1

            # If there was no improvement in the last `max_no_improvement_evals` evaluations, stop training
            if self.no_improvement_count >= self.max_no_improvement_evals:
                self.logger.info("Stopping training due to no improvement.")
                self.model.stop_training = True

    def evaluate_model(self) -> float:
        """
        Evaluate the model over a few episodes and return the average reward.
        Truncate episodes after 100 timesteps if not done.
        :return: The average reward over the evaluation episodes.
        """
        total_reward = 0
        eval_episodes = 5  # Adjust the number of episodes to evaluate
        for _ in range(eval_episodes):
            obs = self.training_env.reset()
            done = False
            episode_reward = 0
            timestep = 0
            while not done and timestep < 100:  # Stop after 100 timesteps
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.training_env.step(action)
                episode_reward += reward
                timestep += 1

            # If the episode isn't done, we truncate the evaluation after 100 timesteps
            if timestep >= 100:
                done = True  # Force the episode to end

            total_reward += episode_reward

        avg_reward = total_reward / eval_episodes
        self.logger.info(f"Evaluation reward: {avg_reward}")
        return avg_reward

    def _on_training_end(self) -> None:
        """Called before exiting the `learn()` method."""
        pass


def learn_envs(environments, max_iterations=10000):
    """
    Trains two models per environment using different reward shaping approaches and saves them.

    Parameters:
    - environments (list of dict): List of environment details.
    - max_iterations (int): Number of training timesteps.

    Returns:
    - dict: Dictionary containing trained models.
    """
    models = {}
    reward_types = ['sparse', 'step_cost']

    for i, Environment in enumerate(environments):
        env_name = Environment['env']
        env_dir = f'data/trained/{env_name}'
        env_logs_dir = f'{env_dir}/Logs'
        env_model_dir = f'{env_dir}/Models'
        print(f"Training on {env_name}")

        n_instructions = 1

        if env_name.startswith('simplest'):
            n_instructions = 1
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining25):
            n_instructions = 1
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining50):
            n_instructions = 2
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining75):
            n_instructions = 3
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining100):
            n_instructions = 4

        # Load game dictionary and room positions
        gameaddress = f'data/Environments/{env_name}'
        game_dict, room_positions = z8file_to_dictionaries(gameaddress)

        for reward_type in reward_types:
            print(f"  Reward Shaping: {reward_type}")
            print(f"  number of Instructions: {n_instructions}")

            # Create and wrap the environment with specified reward_type
            env = TextWorldEnv(
                game_dict=game_dict,
                room_positions=room_positions,
                n_instructions=n_instructions,
                grammar=8,  # Assuming training with 8-sector grammar; adjust if needed
                reward_type=reward_type
            )
            env = Monitor(env, filename=f'{env_logs_dir}/monitor_{reward_type}.log', allow_early_resets=True)

            reward_threshold = 24.9

            # Define callbacks
            callbackOnBest = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
            callbackOnNoImprovement = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=10, verbose=1)
            custom_callback = CustomStopOnNoImprovement(max_no_improvement_evals=3, min_evals=10, verbose=1)
            callback = EvalCallback(
                eval_env=env,
                best_model_save_path=env_model_dir,
                log_path=env_logs_dir,
                eval_freq=20000,
                deterministic=True,
                render=False,
                callback_on_new_best=callbackOnBest,
                n_eval_episodes=10)

            # Initialize or set the model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = DQN('MlpPolicy', env=env, verbose=1, seed=0, device=device, exploration_fraction=0.5,
                        tensorboard_log=f'data/tensorboard',
                        policy_kwargs=dict(net_arch=[256, 256]),
                        learning_rate=0.0005)

            # Learn the model
            model.learn(
                total_timesteps=max_iterations,
                log_interval=20000,
                tb_log_name=f'DQN_{env_name}_{reward_type}',
                reset_num_timesteps=False,
                callback=custom_callback
            )

            # Save the model after training
            model_save_path = f'{env_model_dir}/final_modeldict_{reward_type}.zip'
            model.save(model_save_path)
            print(f"  Saved model to {model_save_path}")

            # Copy the z8 file to the trained folder
            os.makedirs(env_dir, exist_ok=True)
            os.system(f'cp data/Environments/{env_name} {env_dir}/{env_name}')

            # Store the model reference
            models[f"{env_name}_{reward_type}"] = model

    return models

def predict_proba(model, state):
    print(state)
    obs = model.policy.obs_to_tensor(state)[0]
    print(obs)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    print(probs)
    probs_np = probs.detach().cpu().numpy()
    # normalize the probabilities
    probs_np = probs_np / np.sum(probs_np)
    return probs_np

def eval_by_interaction(model, env_info, route_instruction):
    env_name = env_info['env']
    env_dir = f'data/{env_name}'
    env_logs_dir = f'{env_dir}/Logs'

    # Load game dictionary and room positions
    gameaddress = f'data/Environments/{env_name}'
    game_dict, room_positions = z8file_to_dictionaries(gameaddress)

    # Create and wrap the environment
    env = TextWorldEnv(
        game_dict=game_dict,
        room_positions=room_positions,
        x_destination=env_info['x_destination'],
        y_destination=env_info['y_destination'],
        grammar=8,  # Assuming 8-sector grammar; adjust if needed
        reward_type='sparse'  # Adjust if needed
    )
    env = Monitor(env, filename=f'{env_logs_dir}/monitor.log', allow_early_resets=True)

    # Reset the environment
    observation, info = env.reset()
    episode_reward = 0

    # Split the route instruction into sentences
    sentences = route_instruction.split('. ')
    for sentence in sentences:
        action = text_to_action(sentence, env.directions)
        observation, reward, terminate, truncated, _ = env.step(action)
        b = predict_proba(model, observation)
        b = np.round(b, 3)
        print(f"Action: {sentence}, Probability Distribution: {b}")
        prob = b[0][action] if action < len(b[0]) else 0.0

        episode_reward += prob

        print(f"Terminate: {terminate}, Accumulated Probability: {round(episode_reward, 2)}")
        if terminate or truncated:
            print("Terminating the episode")
            break

if __name__ == "__main__":
    # Load the NLP model (ensure it's installed)
    nlp = spacy.load("en_core_web_sm")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Load environments used for pretraining (seen environments)
    all_env_pretraining = load_envs()

    # Learn the environments (training process)
    # learn_envs(all_env_pretraining, max_iterations=1000000)

    # Evaluate all trained models with specified limits
    evaluate_all_trained_models(
        max_seen_envs_per_model=1,        # Limit for seen environments
        max_unseen_envs_per_model=1,      # Limit for unseen environments
        random_seed=42                    # Seed for reproducibility
    )

    # Example evaluation by interaction (optional)
    # environments = get_all_possible_envs()
    # route_instruction = 'go east. go south. go west. go southwest. go southwest. Arrive at destination!'
    # another_route_instruction = 'go west'
    # env_info = environments[0]  # Example
    # eval_by_interaction(model, env_info, route_instruction)