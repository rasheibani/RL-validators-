Environment = 'data/Environments/A_Average-Regular_Approach1_545604.9376088154_1000842.9379071898.z8'
RI = 'go east. go south. go west. go southwest. go southwest. Arrive at destination!'


import textworld.gym # Register the gym environments
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
import re
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3.common.env_util import make_vec_env



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

def get_admissible_actions(feedback):
    directions = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']
    admissible_actions = []

    for direction in directions:
        pattern = r'going ' + direction + ' '
        if re.search(pattern, feedback, re.IGNORECASE):
            admissible_actions.append('go '+direction)

    return admissible_actions
def feedback_to_embedding(feedback):
    doc = nlp(feedback)
    embeddings = []
    for token in doc:
        embeddings.append(token.vector)

    if embeddings:
        return np.mean(embeddings, axis=0).astype(np.float32)
    else:
        return np.zeros((300,), dtype=np.float32)




class TextWorldEnv(gym.Env):
    def __init__(self, game_address, total_steps=10000):
        self.y = None
        self.x = None
        self.is_async = False
        self.game_address = game_address
        self.env = textworld.start(game_address)
        self.game_state = None
        self.action_space = spaces.Discrete(8)
        self.visited_states_actions = set()
        self.last_feedback_embedding = None
        self.counter = 0

        # The observation space is a Dict space with the following keys:
        # - 'text': a string representing the current game state
        # - 'x': the x-coordinate of the player
        # - 'y': the y-coordinate of the player
        self.observation_space = spaces.Dict({
            'text': spaces.Box(low=-np.inf, high=np.inf, shape=(96,), dtype='float32')})

    def reset(self, **kwargs):
        self.game_state = self.env.reset()
        self.x, self.y = extract_coordinates(self.game_state.feedback)
        x_as_array = np.array([self.x], dtype='float32')
        y_as_array = np.array([self.y], dtype='float32')
        feedback_embedding = feedback_to_embedding(self.game_state.feedback)
        self.last_feedback_embedding = feedback_embedding
        observation = {'text': feedback_embedding }
        info = {}  # Create an empty info dictionary
        self.visited_states_actions.clear()
        return observation, info

    def step(self, action):
        # check if the action is within the admissible actions
        admissible_actions = get_admissible_actions(self.game_state.feedback)
        # print(self.game_state.feedback)
        # print(admissible_actions)

        sentence = ""
        if int(action) == int(0):
            sentence = "go north"
        elif int(action) == int(1):
            sentence = "go south"
        elif int(action) == int(2):
            sentence = "go east"
        elif int(action) == int(3):
            sentence = "go west"
        elif int(action) == int(4):
            sentence = "go northeast"
        elif int(action) == int(5):
            sentence = "go northwest"
        elif int(action) == int(6):
            sentence = "go southeast"
        elif int(action) == int(7):
            sentence = "go southwest"

        if sentence not in admissible_actions:
            # print(f"Action {sentence} is not admissible. Skipping.")
            reward = -1
            terminate = False
            truncated = False
            observation = {'text': self.last_feedback_embedding}
            return observation, reward, terminate, truncated, {}
        else:
            self.game_state, reward, done_dummy = self.env.step(sentence)
            self.x, self.y = extract_coordinates(self.game_state.feedback)
            feedback_embedding = feedback_to_embedding(self.game_state.feedback)
            self.last_feedback_embedding = feedback_embedding
            target_x = 546281.0
            target_y = 999991.0
            distance = np.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)

            if np.isclose(self.x, target_x, atol=1e-3) and np.isclose(self.y, target_y, atol=1e-3):
                reward = 500
                terminate = True
                # print(self.game_state.feedback)
                self.counter = self.counter + 1
                # print(self.counter)

                # print('----------------------------------------------------------------------')
                # print('---------Destination reached------------------------------------------')

            else:
                reward = -0.1
                terminate = False
            truncated = False
            self.visited_states_actions.add((self.x, self.y, action))
            # if self.counter > 0:
            #     print(f"X: {self.x}, Y: {self.y}, Action: {action}, Reward: {reward}")

            x_as_array = np.array([self.x], dtype='float32')
            y_as_array = np.array([self.y], dtype='float32')
            observation = {'text': feedback_embedding}


            return observation, reward, terminate, truncated, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __len__(self):
        return 1  # only one game running at a time
    def extract_area_id(self,feedback):
        pattern = r"An area \((\d+)\) in r(\d+)"
        matches = re.search(pattern, feedback)
        if matches:
            area_id = matches.group(1)
            room_id = matches.group(2)
            return f"a{area_id}r{room_id}"
        else:
            return None



if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    env = TextWorldEnv(Environment)
    model = PPO("MultiInputPolicy", env, verbose=1, seed=0)
    model.learn(total_timesteps=1000000, log_interval=1)


