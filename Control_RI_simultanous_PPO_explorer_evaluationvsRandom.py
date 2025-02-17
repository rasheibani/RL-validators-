import random
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
import re

from pydantic.v1.utils import truncate
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import os
from z8file_to_dictionaries import z8file_to_dictionaries
from tqdm import tqdm  # For progress bars
import warnings

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
    Converts a text instruction (e.g., "go north") to an action index.
    Extracts the direction word from the instruction.
    """
    mapping = {direction: idx for idx, direction in enumerate(directions)}

    words = text.strip().lower().split()  # Split the instruction into words
    direction_word = None
    for word in words:
        if word in mapping:  # Check if any word is a valid direction
            direction_word = word
            break # Assuming only one direction word per instruction

    if direction_word:
        return mapping[direction_word]
    else:
        return -1  # Invalid action

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
    def __init__(self, game_dict, room_positions, x_destination=None, y_destination=None, n_instructions=1, grammar=8,
                 reward_type='sparse'):
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
            # make sure the origin and destination are not the same
            if temp_room_id != self.current_room_id:
                print("same room for origin and destination")
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
            # check if the x y of destination is None, then find them from the route instructions
            if self.x_destination is None or self.y_destination is None:
                self.x_destination, self.y_destination = self.get_destination_from_route_instructions(self.route_instructions)
        else:
            self.route_instructions, self.x_destination, self.y_destination = self.generate_route_instructions()
            # print(f"Route instructions: {self.route_instructions}")

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
        global reward
        sentence = self.sentence_from_action_func(action)
        admissible_actions = self.get_admissible_actions()
        terminate = False
        truncated = False
        self.counter += 1


        # Define "look" action
        look_action_index = self.max_directions
        if action == look_action_index:
            if self.reward_type == 'sparse':
                reward = 0
            elif self.reward_type == 'step_cost':
                reward = -1  # Step cost
            terminate = False
            truncated = False
            observation = self.construct_observation(admissible_actions)
            return observation, reward, terminate, truncated, {}

        if sentence not in admissible_actions:
            if self.reward_type == 'sparse':
                reward = 0
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

            if np.isclose(self.x, target_x, atol =1e-2) and np.isclose(self.y, target_y, atol=1e-2):
                # check if they are exactly the same then the reward is zero
                if self.x == target_x and self.y == target_y:
                    reward = 0
                    terminate = False
                    truncated = False
                else:
                    reward = 25
                    terminate = True
                    truncated = False
                observation = self.construct_observation(admissible_actions)
                # print("terminate1", terminate)
            else:
                if self.reward_type == 'step_cost':
                    # stop long exploration (after 30 steps)
                    reward = -1
                    if self.counter > self.n_instructions + self.exploration_threshold:
                        reward = -1
                        terminate = False
                        truncated = True
                        observation = self.construct_observation(admissible_actions)
                        return observation, reward, terminate, truncated, {}
                elif self.reward_type == 'sparse':
                    reward = 0
                    # Check for step limit
                    if self.instruction_index >= len(self.route_instructions) + self.exploration_threshold - 1:
                        reward = -1
                        terminate = False
                        truncated = True
                        observation = self.construct_observation(admissible_actions)
                        self.instruction_index += 1
                        return observation, reward, terminate, truncated, {}

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

    # load 'for complexity' file, there is a column named 'distinct' witch env_names start with it
    # also there is a column named 'Normalized Categorized Bearing Entropy' witch is the complexity of the env

    with open('data/for_complexity.csv', 'r') as f:
        df = pd.read_csv(f)
        for i in range(len(df)):
            # print ( ' i = ', i, 'env_name = ', env_name, 'df[distinct][i] = ', df['distinct'][i]) if env_name.startswith(df['distinct'][i]) else None
            if env_name.startswith(df['distinct'][i]):

                return df['Normalized Bearing Entropy'][i]
            # if you don't find the env_name in the file, return 0 if the env_name starts with 'simplest' or has only one character
            # before first '_' , return 0.26 if two characters before first '_' and, 0.76 if threecharacters before first '_'

        if env_name.startswith('simplest'):
            return 0
        elif len(env_name.split('_')[0]) <= 1:
            return 0
        elif len(env_name.split('_')[0]) == 2:
            return 0.27
        elif len(env_name.split('_')[0]) >= 3:
            return 0.52
        else:
            return 0.77

    return 1000  # Default value



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
    if env_name.startswith('simplest'):
        n_instructions = 1
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining25):
        n_instructions = 2
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining50):
        n_instructions = 3
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining75):
        n_instructions = 4
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining100):
        n_instructions = 5
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
        reward = 0

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

        successes.append(reward)

        if verbose:
            print(f"Episode {episode + 1}:")
            print(f"  Actions Taken: {episode_actions}")
            print(f"  Distances: {episode_distances}")
            print(f"  Success: {reward}")


    return successes

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


def evaluate_curriculum_models(max_seen_envs=5, max_unseen_envs=5, random_seed=42):
    """
    Evaluates the final curriculum models (sparse and step-cost) on seen/unseen environments
    with complete/incomplete instructions across all grammars, comparing against random agents.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Define evaluation parameters
    grammars = [8, 6, 4]
    reward_types = ['sparse', 'step_cost']
    results = []

    # Load environment data
    all_env_pretraining = load_envs()
    seen_env_names = {env['env'] for env in all_env_pretraining}
    all_envs = get_all_possible_envs()
    unseen_envs = [env for env in all_envs if env['env'] not in seen_env_names]

    # Select subset of environments
    selected_seen = random.sample(all_env_pretraining, min(max_seen_envs, len(all_env_pretraining)))
    selected_unseen = random.sample(unseen_envs, min(max_unseen_envs, len(unseen_envs)))
    eval_envs = selected_seen + selected_unseen

    for reward_type in reward_types:
        print(f"\n=== Evaluating {reward_type} reward model ===")

        # Load curriculum model
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

        # Evaluate across all grammars and environments
        for grammar in grammars:
            print(f"\nEvaluating grammar {grammar}")

            for env_info in tqdm(eval_envs, desc="Environments"):
                env_name = env_info['env']
                is_seen = 'seen' if env_name in seen_env_names else 'unseen'

                # Load game environment
                try:
                    game_dict, room_positions = z8file_to_dictionaries(f'data/Environments/{env_name}')
                except Exception as e:
                    print(f"Skipping {env_name}: {e}")
                    continue

                # Create environment instance
                env = TextWorldEnv(
                    game_dict=game_dict,
                    room_positions=room_positions,
                    n_instructions=3,
                    grammar=grammar,
                    reward_type=reward_type
                )
                env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

                # Evaluate complete instructions
                success_trained, std_trained = evaluate_model(model, env)

                # Evaluate random agent
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

                # Evaluate incomplete instructions
                if len(env.route_instructions) >= 3:
                    env_incomplete = create_incomplete_environment(env, game_dict, room_positions, grammar, reward_type)
                    success_trained_inc, std_trained_inc = evaluate_model(model, env_incomplete)

                    # Evaluate random agent on incomplete instructions
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

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('data/curriculum_evaluation_results.csv', index=False)
    print("\nEvaluation complete. Results saved to curriculum_evaluation_results.csv")
    return df

# Helper functions
def evaluate_model(model, env, n_episodes=10):
    try:
        evaluation_result = evaluate_policy( # Store the result, don't unpack immediately
            model.policy, env, n_eval_episodes=n_episodes,
            deterministic=True, warn=False, return_episode_rewards=False # Important: Ensure return_episode_rewards=False (default, but explicit is good)
        )

        if not isinstance(evaluation_result, tuple) or len(evaluation_result) != 2: # Type check the return

            mean_success = np.nan  # Assign NaN values to indicate failure
            std_success = np.nan
        else: # Expected tuple return
            rewards, _ = evaluation_result # Now unpack if it's a tuple (though SB3 should return tuple of mean, std)
            if not isinstance(rewards, (int, float, np.number)): # Check if 'rewards' is actually mean_reward (SB3 returns mean, std tuple)
                 print(f"!!! WARNING: Unexpected type for 'rewards' (mean reward): {type(rewards)}, expected number")
                 mean_success = np.nan
                 std_success = np.nan
            else:

                successes = [] # Recalculate successes based on individual episode rewards (if you had them, but you don't in this setup anymore) -  No longer recalculating successes directly from evaluate_policy output as we are expecting mean/std already.
                mean_success = rewards # SB3 evaluate_policy returns mean reward as first element of tuple. Rename for clarity
                std_success = evaluation_result[1] # Standard deviation is the second element

        return mean_success, std_success

    except Exception as e:
        print(f"!!! Exception in evaluate_model !!!: {e}") # Debugging - Exception occurred
        print(f"Returning NaN, NaN from except block...") # Debugging - returning from except
        return np.nan, np.nan

def create_incomplete_environment(base_env, game_dict, room_positions, grammar, reward_type):
    """Create environment with incomplete instructions"""
    if len(base_env.route_instructions) < 3:
        return None

    # Remove middle instruction
    incomplete_instructions = base_env.route_instructions.copy()
    del incomplete_instructions[len(incomplete_instructions) // 2]

    # Create new environment
    env = TextWorldEnv(
        game_dict=game_dict,
        room_positions=room_positions,
        n_instructions=4,
        grammar=grammar,
        reward_type=reward_type
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    # Convert instructions to text format
    instruction_text = '. '.join(
        [sentence_from_action(a, env.directions) for a in incomplete_instructions]
    ) + '. Arrive at destination!'

    env.reset(route_instructions=instruction_text)

    # overwrite the x_destination and y_destination from the base_env
    env.x_destination, env.y_destination = base_env.x_destination, base_env.y_destination
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
    Trains models using curriculum learning, loading the best model from previous environments.
    """
    models = {}
    reward_types = ['sparse', 'step_cost']

    # Sort environments by complexity
    sorted_envs = sorted(
        environments,
        key=lambda x: determine_complexity(x['env'])
    )

    for reward_type in reward_types:
        print(f"\n=== Training with reward type: {reward_type} ===")
        model = None
        curriculum_model_dir = f'data/trained/curriculum/PPO/{reward_type}'
        os.makedirs(curriculum_model_dir, exist_ok=True)
        model_path = os.path.join(curriculum_model_dir, 'best_model.zip')

        for env_info in sorted_envs:
            env_name = env_info['env']
            print(f"\nTraining on environment: {env_name}")

            # Load game dictionary and room positions
            gameaddress = f'data/Environments/{env_name}'
            try:
                game_dict, room_positions = z8file_to_dictionaries(gameaddress)
            except Exception as e:
                print(f"Skipping {env_name} due to error: {e}")
                continue

            # Create and wrap environment
            env = TextWorldEnv(
                game_dict=game_dict,
                room_positions=room_positions,
                n_instructions=determine_n_instructions(env_name),
                grammar=8,
                reward_type=reward_type
            )
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
            env = Monitor(env, filename=f'{curriculum_model_dir}/monitor_{env_name}.log')

            # Load or initialize model
            if os.path.exists(model_path):
                print(f"Loading previous best model from {model_path}")
                model = PPO.load(model_path, env=env, device='auto')
                model.set_env(env)
            else:
                print("Initializing new model")
                model = PPO(
                    'MlpPolicy', env=env,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    verbose=0,
                    policy_kwargs=dict(net_arch=[64, 64]),
                    learning_rate=0.001,
                    tensorboard_log='data/tensorboard'
                )

            # Configure callbacks
            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=24.9, verbose=1)
            eval_callback = EvalCallback(
                env,
                best_model_save_path=curriculum_model_dir,
                eval_freq=5000,
                callback_on_new_best=stop_callback,
                verbose=0
            )

            # Train the model
            model.learn(
                total_timesteps=max_iterations,
                callback=eval_callback,
                tb_log_name=f"PPO_curriculum_{reward_type}",
                reset_num_timesteps=False
            )

            # Save the final model after environment training
            model.save(model_path)
            print(f"Saved trained model to {model_path}")

        models[reward_type] = model

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
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load environments used for pretraining (seen environments)
    all_env_pretraining = load_envs()

    # Learn the environments (training process)
    # learn_envs(all_env_pretraining, max_iterations=1000000)

    # Evaluate all trained models with specified limits
    evaluate_curriculum_models(
        max_seen_envs=30,
        max_unseen_envs=30,
        random_seed=2                 # Seed for reproducibility
    )

    # Example evaluation by interaction (optional)
    # environments = get_all_possible_envs()
    # route_instruction = 'go east. go south. go west. go southwest. go southwest. Arrive at destination!'
    # another_route_instruction = 'go west'
    # env_info = environments[0]  # Example
    # eval_by_interaction(model, env_info, route_instruction)