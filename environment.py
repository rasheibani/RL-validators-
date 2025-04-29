import gymnasium as gym
from gymnasium import spaces
import numpy as np
import re
import random

# Define grammars and their corresponding directions
GRAMMAR_DIRECTIONS = {
    4: ['north', 'south', 'east', 'west'],
    6: ['north', 'south', 'east', 'west', 'northeast', 'southwest'],
    8: ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']
}

# Maximum distance for normalization (adjust based on your environment)
MAX_DISTANCE = 2000.0

def extract_coordinates(game_state):
    pattern = r"X:\s*([\d.]+)\s*\nY:\s*([\d.]+)"
    matches = re.search(pattern, game_state)
    if matches:
        x = float(matches.group(1))
        y = float(matches.group(2))
        return np.array([x, y])
    return np.array([0, 0])

def text_to_action(text, directions):
    mapping = {direction: idx for idx, direction in enumerate(directions)}
    words = text.strip().lower().split()
    direction_word = next((word for word in words if word in mapping), None)
    return mapping[direction_word] if direction_word else -1

def sentence_from_action(action, directions):
    if 0 <= action < len(directions):
        return f"go {directions[action]}"
    return "look"

def normalize(observation):
    max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
    admissible_actions = observation[:max_directions]
    route_instructions = observation[max_directions:-2]
    instruction_indices = observation[-2:]

    normalized_admissible_actions = admissible_actions
    normalized_route_instructions = np.where(
        route_instructions != max_directions,
        route_instructions / (max_directions - 1),
        -1
    )
    max_instruction_index = len(route_instructions)
    normalized_indices = np.array([
        instruction_indices[0] / max(max_instruction_index, 1),
        instruction_indices[1] / (max_directions - 1) if instruction_indices[1] != max_directions else -1
    ])
    return np.concatenate([
        normalized_admissible_actions,
        normalized_route_instructions,
        normalized_indices
    ])

def get_admissible_actions(feedback, directions):
    admissible_actions = []
    for direction in directions:
        pattern = r'\bgoing ' + direction + r'\b'
        if re.search(pattern, feedback, re.IGNORECASE):
            admissible_actions.append('go ' + direction)
    return admissible_actions

def admissible_actions_to_observation(admissible_actions, directions):
    max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
    observation = np.zeros(max_directions, dtype=np.int32)
    for i, direction in enumerate(directions):
        if f"go {direction}" in admissible_actions:
            observation[i] = 1
    return observation

def extract_area_id(feedback):
    pattern = r"An area $$(\d+)$$ in r(\d+)"
    matches = re.search(pattern, feedback)
    if matches:
        area_id = matches.group(1)
        room_id = matches.group(2)
        return f"a{area_id}r{room_id}"
    print("Failed to extract area ID from feedback:")
    print(feedback)
    print("-" * 50)
    return None

class TextWorldEnv(gym.Env):
    def __init__(self, game_dict, room_positions, x_destination=None, y_destination=None, n_instructions=1, grammar=8,
                 reward_type='sparse', is_incomplete=False, route_instructions=None):
        super(TextWorldEnv, self).__init__()
        self.game_dict = game_dict
        self.room_positions = room_positions
        self.current_room_id = None
        self.n_instructions = n_instructions
        self.grammar = grammar
        self.reward_type = reward_type
        self.instruction_index = 0
        self.route_instructions = [] if route_instructions is None else route_instructions
        self.visited_states_actions = set()
        self.last_feedback_embedding = None
        self.counter = 0
        self.is_incomplete = is_incomplete
        self.x_destination = x_destination
        self.y_destination = y_destination
        self.x_origin = None
        self.y_origin = None
        self.exploration_threshold = 0

        if self.grammar in GRAMMAR_DIRECTIONS:
            self.directions = GRAMMAR_DIRECTIONS[self.grammar]
        else:
            raise ValueError("Invalid grammar. Choose from 4, 6, or 8.")

        self.max_directions = max(len(dirs) for dirs in GRAMMAR_DIRECTIONS.values())
        self.action_space = spaces.Discrete(self.max_directions + 1)
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
                break
            temp_room_id = next_room_id
            if temp_room_id != self.current_room_id:
                print("same room for origin and destination")
        return self.room_positions[temp_room_id]

    def reset(self, **kwargs):
        self.counter = 0
        self.current_room_id = random.choice(list(self.game_dict.keys()))
        self.x, self.y = self.room_positions[self.current_room_id]
        self.x_origin, self.y_origin = self.x, self.y
        self.visited_states_actions.clear()
        self.instruction_index = 0

        if self.is_incomplete:
            if self.x_destination is None or self.y_destination is None:
                raise ValueError("x_destination and y_destination must be provided for incomplete instructions")
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
        assert observation.shape[0] == self.observation_space.shape[0], \
            f"Observation shape mismatch. Expected {self.observation_space.shape[0]}, got {observation.shape[0]}"

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
                constant_values=self.max_directions
            )
        return self.route_instructions[:15]

    def construct_observation(self, admissible_actions):
        current_instruction = (
            self.route_instructions[self.instruction_index]
            if self.instruction_index < len(self.route_instructions)
            else self.max_directions
        )
        observation = np.concatenate([
            admissible_actions_to_observation(admissible_actions, self.directions),
            self.pad_instructions(),
            np.array([self.instruction_index, current_instruction])
        ])
        expected_shape = self.observation_space.shape[0]
        assert observation.shape[0] == expected_shape, \
            f"Observation shape mismatch. Expected {expected_shape}, got {observation.shape[0]}"
        return normalize(observation)

    def step(self, action):
        global reward
        sentence = self.sentence_from_action_func(action)
        admissible_actions = self.get_admissible_actions()
        terminate = False
        truncated = False
        self.counter += 1

        look_action_index = self.max_directions
        if action == look_action_index:
            if self.reward_type == 'sparse':
                reward = 0
            elif self.reward_type == 'step_cost':
                reward = -1
            terminate = False
            truncated = False
            observation = self.construct_observation(admissible_actions)
            return observation, reward, terminate, truncated, {}

        if sentence not in admissible_actions:
            if self.reward_type == 'sparse':
                reward = 0
            elif self.reward_type == 'step_cost':
                reward = -1
            terminate = False
            truncated = False
            observation = self.construct_observation(admissible_actions)
            return observation, reward, terminate, truncated, {}
        else:
            next_room_id = self.game_dict[self.current_room_id][sentence]
            if next_room_id is None:
                if self.reward_type == 'sparse':
                    reward = -1
                elif self.reward_type == 'step_cost':
                    reward = -1
                terminate = False
                truncated = False
                observation = self.construct_observation(admissible_actions)
                return observation, reward, terminate, truncated, {}

            self.current_room_id = next_room_id
            self.x, self.y = self.room_positions[self.current_room_id]

            target_x = self.x_destination
            target_y = self.y_destination

            if np.isclose(self.x, target_x, atol=1e-2) and np.isclose(self.y, target_y, atol=1e-2):
                if self.x == target_x and self.y == target_y:
                    reward = 0
                    terminate = False
                    truncated = False
                else:
                    reward = 25
                    terminate = True
                    truncated = False
                observation = self.construct_observation(admissible_actions)
            else:
                if self.reward_type == 'step_cost':
                    reward = -1
                    if self.counter > self.n_instructions + self.exploration_threshold:
                        reward = -1
                        terminate = False
                        truncated = True
                        observation = self.construct_observation(admissible_actions)
                        return observation, reward, terminate, truncated, {}
                elif self.reward_type == 'sparse':
                    reward = 0
                    if self.instruction_index >= len(self.route_instructions) + self.exploration_threshold - 1:
                        reward = -1
                        terminate = False
                        truncated = True
                        observation = self.construct_observation(admissible_actions)
                        self.instruction_index += 1
                        return observation, reward, terminate, truncated, {}

            admissible_actions = self.get_admissible_actions()
            observation = self.construct_observation(admissible_actions)
            self.instruction_index += 1
            self.visited_states_actions.add((self.current_room_id, sentence))
            return observation, reward, terminate, truncated, {}

    def render(self):
        pass

    def close(self):
        pass

    def __len__(self):
        return 1
