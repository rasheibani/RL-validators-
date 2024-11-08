import random
from asyncio import current_task
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
import re
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import multiprocessing
import Pretraining
from stable_baselines3.common.policies import obs_as_tensor
import xml.etree.ElementTree as ET
import os
from z8file_to_dictionaries import z8file_to_dictionaries
from tqdm import tqdm  # For progress bars

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
    # Separate the components
    admissible_actions = observation[:len(GRAMMAR_DIRECTIONS[8])]  # Adjust based on maximum directions
    route_instructions = observation[len(GRAMMAR_DIRECTIONS[8]):-1]  # Next part are route instructions
    instruction_index = observation[-1]  # Last is instruction index

    # Normalize admissible actions (already in [0, 1])
    normalized_admissible_actions = admissible_actions

    # Normalize route instructions, treating 8 as padding and replacing it with -1
    # Adjust padding value based on the maximum number of directions
    max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
    normalized_route_instructions = np.where(route_instructions != max_directions, route_instructions / (max_directions - 1), -1)

    # Normalize instruction index
    max_instruction_index = len(route_instructions)
    normalized_instruction_index = instruction_index / max_instruction_index if max_instruction_index > 0 else 0

    # Combine normalized components
    normalized_observation = np.concatenate(
        [normalized_admissible_actions, normalized_route_instructions, [normalized_instruction_index]]
    )

    return normalized_observation

def get_admissible_actions(feedback, directions):
    admissible_actions = []
    for direction in directions:
        pattern = r'going ' + direction + ' '
        if re.search(pattern, feedback, re.IGNORECASE):
            admissible_actions.append('go ' + direction)
    return admissible_actions

def admissible_actions_to_observation(admissible_actions, directions):
    observation = np.zeros(len(directions), dtype=np.int32)
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
    def __init__(self, game_dict, room_positions, x_destination=None, y_destination=None, n_instructions=5, grammar=8):
        super(TextWorldEnv, self).__init__()
        self.game_dict = game_dict  # The game dictionary
        self.room_positions = room_positions  # Mapping from room IDs to (x, y) coordinates
        self.current_room_id = None
        self.n_instructions = n_instructions
        self.grammar = grammar  # 4, 6, or 8
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

        self.action_space = spaces.Discrete(len(self.directions) + 1)  # +1 for "look"
        self.observation_space = spaces.Box(low=0, high=8, shape=(24,), dtype=np.int32)

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
        # Set the current room to a random starting room
        self.current_room_id = random.choice(list(self.game_dict.keys()))
        self.x, self.y = self.room_positions[self.current_room_id]
        self.x_origin, self.y_origin = self.x, self.y
        self.visited_states_actions.clear()
        self.instruction_index = 0

        if 'route_instructions' in kwargs:
            rti = kwargs['route_instructions']
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
            np.array([self.instruction_index])
        ))
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
                constant_values=len(self.directions)  # Use len(directions) as padding value
            )
        else:
            return self.route_instructions[:15]

    def construct_observation(self, admissible_actions):
        observation = np.concatenate((
            admissible_actions_to_observation(admissible_actions, self.directions),
            self.pad_instructions(),
            np.array([self.instruction_index])
        ))
        observation = normalize(observation)
        return observation

    def step(self, action):
        sentence = self.sentence_from_action_func(action)
        admissible_actions = self.get_admissible_actions()

        # Define "look" action
        look_action_index = len(self.directions)
        if action == look_action_index:
            reward = -1
            terminate = False
            truncated = False
            observation = self.construct_observation(admissible_actions)
            return observation, reward, terminate, truncated, {}

        if sentence not in admissible_actions:
            reward = -1
            terminate = False
            truncated = False
            observation = self.construct_observation(admissible_actions)
            return observation, reward, terminate, truncated, {}
        else:
            # Move to the next room
            next_room_id = self.game_dict[self.current_room_id][sentence]
            if next_room_id is None:
                reward = -1
                terminate = False
                truncated = False
                observation = self.construct_observation(admissible_actions)
                return observation, reward, terminate, truncated, {}

            self.current_room_id = next_room_id
            self.x, self.y = self.room_positions[self.current_room_id]

            # Calculate reward
            target_x = self.x_destination
            target_y = self.y_destination

            if np.isclose(self.x, target_x, atol=1e-3) and np.isclose(self.y, target_y, atol=1e-3):
                reward = 25
                terminate = True
                truncated = False
            else:
                if self.instruction_index >= len(self.route_instructions) + self.exploration_threshold - 1:
                    reward = -1
                    terminate = False
                    truncated = True
                    observation = self.construct_observation(admissible_actions)
                    self.instruction_index += 1
                    return observation, reward, terminate, truncated, {}
                reward = 0
                terminate = False
                truncated = False

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
        n_instructions = 4
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining75):
        n_instructions = 7
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining100):
        n_instructions = 10
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
                action = len(env.directions)  # Assuming "look" is the last action
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
    # Assuming that 'distance' is the last element of the observation
    # Adjust the index based on your observation structure
    distance = observation[-1] * MAX_DISTANCE  # Replace MAX_DISTANCE with your scaling factor
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
        model_path = f'data/trained/{model_folder}/Models/final_modeldict.zip'

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
            for env_info in tqdm(eval_envs, desc=f"Grammar {grammar} - Evaluating Envs for Model {model_folder}", leave=False):
                env_name = env_info['env']
                is_seen = 'seen' if env_name in seen_env_names else 'unseen'

                # Determine the complexity of the environment
                complexity = determine_complexity(env_name)
                complexity = assign_complexity_category_updated(complexity)

                # Load game dictionary and room positions
                gameaddress = f'data/Environments/{env_name}'
                try:
                    game_dict, room_positions = z8file_to_dictionaries(gameaddress)
                except Exception as e:
                    print(f"Failed to load game dictionary from {gameaddress}: {e}")
                    continue

                # Create the TextWorldEnv with the specified grammar
                env = TextWorldEnv(
                    game_dict=game_dict,
                    room_positions=room_positions,
                    n_instructions=15,  # Adjust as needed
                    grammar=grammar
                )
                env_logs_dir = f'data/trained/{model_folder}/Logs'
                os.makedirs(env_logs_dir, exist_ok=True)
                env = Monitor(env, filename=f'{env_logs_dir}/monitor.log', allow_early_resets=True)

                # Evaluate the trained model
                try:
                    episode_rewards, _ = evaluate_policy(
                        model,
                        env,
                        n_eval_episodes=100,
                        deterministic=False,
                        render=False,
                        callback=None,
                        reward_threshold=None,
                        return_episode_rewards=True,
                        warn=False
                    )
                    successes = [1 if reward >= 25 else 0 for reward in episode_rewards]
                    average_success_rate = sum(successes) / len(successes)
                    std_success_rate = np.std(successes)
                except Exception as e:
                    print(f"Failed to evaluate model {model_folder} on environment {env_name} with grammar {grammar}: {e}")
                    continue

                # Evaluate the random agent on the same environment
                try:
                    # Create a fresh environment instance for the random agent with the same grammar
                    random_env = TextWorldEnv(
                        game_dict=game_dict,
                        room_positions=room_positions,
                        n_instructions=15,  # Must match the main env
                        grammar=grammar
                    )
                    random_env = Monitor(random_env, filename=f'{env_logs_dir}/random_agent_monitor.log', allow_early_resets=True)

                    random_average_success_rate, random_std_success_rate = evaluate_random_agent(
                        random_env,
                        n_eval_episodes=100,
                        verbose=False
                    )
                except Exception as e:
                    print(f"Failed to evaluate random agent on environment {env_name} with grammar {grammar}: {e}")
                    random_average_success_rate = np.nan
                    random_std_success_rate = np.nan

                # Append the results to the list
                results.append({
                    'name_of_env': env_name,
                    'complexity': complexity,
                    'grammar': grammar,
                    'average_success_rate': average_success_rate,
                    'std_success_rate': std_success_rate,
                    'random_agent_average_success_rate': random_average_success_rate,
                    'random_agent_std_success_rate': random_std_success_rate,
                    'evaluated_env': is_seen
                })

                print(f"Model {model_folder} evaluated on environment {env_name} with grammar {grammar} ({is_seen})")
                print(f"Random Agent - Average Success Rate: {random_average_success_rate}, Std: {random_std_success_rate}")

def learn_envs(environments, max_iterations=10000):
    model = None
    for i, Environment in enumerate(environments):
        env_name = Environment['env']
        env_dir = f'data/trained/{env_name}'
        env_logs_dir = f'{env_dir}/Logs'
        env_model_dir = f'{env_dir}/Models'
        print(f"Training on {env_name}")

        n_instructions = 1

        if env_name == 'simplest_simplest_546025.6070834016_1005814.3004094235.z8':
            n_instructions = 1
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining25):
            n_instructions = 2
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining50):
            n_instructions = 4
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining75):
            n_instructions = 7
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining100):
            n_instructions = 10

        # Load game dictionary and room positions
        gameaddress = f'data/Environments/{env_name}'
        game_dict, room_positions = z8file_to_dictionaries(gameaddress)

        # Create and wrap the environment
        env = TextWorldEnv(
            game_dict=game_dict,
            room_positions=room_positions,
            n_instructions=n_instructions,
            grammar=8  # Assuming training with 8-sector grammar; adjust if needed
        )
        env = Monitor(env, filename=f'{env_logs_dir}/monitor.log', allow_early_resets=True)

        reward_threshold = 19

        callbackOnBest = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        callbackOnNoImprovement = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=10, verbose=1)
        callback = EvalCallback(
            eval_env=env,
            best_model_save_path=env_model_dir,
            log_path=env_logs_dir,
            eval_freq=50000,
            deterministic=False,
            render=False,
            callback_on_new_best=callbackOnBest
        )

        # Load model if it exists, otherwise initialize it
        if model is None:
            model = DQN('MlpPolicy', env=env, verbose=1, seed=0, device='cuda' if torch.cuda.is_available() else 'cpu', exploration_fraction=0.99)
        else:
            model.set_env(env)

        # Learn the model
        model.learn(
            total_timesteps=max_iterations,
            log_interval=50000,
            tb_log_name=f'DQN_{env_name}',
            reset_num_timesteps=True,
            callback=callback
        )

        # Save the model after training
        model.save(f'{env_model_dir}/final_modeldict')
        # Copy the z8 file to the trained folder
        os.system(f'cp data/Environments/{env_name} {env_dir}/{env_name}')

    return model

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
        grammar=8  # Assuming 8-sector grammar; adjust if needed
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
    nlp = spacy.load("en_core_web_sm")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Load environments used for pretraining (seen environments)
    all_env_pretraining = load_envs()

    # Learn the environments (training process)
    # model = learn_envs(all_env_pretraining, max_iterations=100000)

    # Evaluate all trained models with specified limits
    evaluate_all_trained_models(
        max_seen_envs_per_model=1,        # Limit for seen environments
        max_unseen_envs_per_model=1,      # Limit for unseen environments
        random_seed=42                     # Seed for reproducibility
    )

    # Example evaluation by interaction (optional)
    # environments = get_all_possible_envs()
    # route_instruction = 'go east. go south. go west. go southwest. go southwest. Arrive at destination!'
    # another_route_instruction = 'go west'
    # env_info = environments[0]  # Example
    # eval_by_interaction(model, env_info, route_instruction)
