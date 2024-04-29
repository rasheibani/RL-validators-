Environment = 'data/Environments/A_Average-Regular_Approach1_545604.9376088154_1000842.9379071898.z8'
RI = 'go east. go south. go west. go southwest. go southwest. Arrive at destination!'

import textworld.gym  # Register the gym environments
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
import re
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement

from stable_baselines3.common.monitor import Monitor
import multiprocessing

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
            admissible_actions.append('go ' + direction)

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
    def __init__(self, game_address, x_destination, y_destination):
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
        self.x_destination = x_destination
        self.y_destination = y_destination

        # The observation space is a Dict space with the following keys:
        # - 'text': a string representing the current game state
        # - 'x': the x-coordinate of the player
        # - 'y': the y-coordinate of the player
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(96,), dtype='float32')

    def reset(self, **kwargs):
        self.game_state = self.env.reset()
        self.x, self.y = extract_coordinates(self.game_state.feedback)
        # x_as_array = np.array([self.x], dtype='float32')
        # y_as_array = np.array([self.y], dtype='float32')
        feedback_embedding = feedback_to_embedding(self.game_state.feedback)
        self.last_feedback_embedding = feedback_embedding
        observation = feedback_embedding
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
            observation = self.last_feedback_embedding
            return observation, reward, terminate, truncated, {}
        else:
            self.game_state, reward, done_dummy = self.env.step(sentence)
            self.x, self.y = extract_coordinates(self.game_state.feedback)
            feedback_embedding = feedback_to_embedding(self.game_state.feedback)
            self.last_feedback_embedding = feedback_embedding
            # convert self.x_destination to numpy.float64
            target_x = np.float64(self.x_destination)
            target_y = np.float64(self.y_destination)
            # distance = np.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)

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

            # x_as_array = np.array([self.x], dtype='float32')
            # y_as_array = np.array([self.y], dtype='float32')
            observation = feedback_embedding

            return observation, reward, terminate, truncated, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __len__(self):
        return 1  # only one game running at a time

    def extract_area_id(self, feedback):
        pattern = r"An area \((\d+)\) in r(\d+)"
        matches = re.search(pattern, feedback)
        if matches:
            area_id = matches.group(1)
            room_id = matches.group(2)
            return f"a{area_id}r{room_id}"
        else:
            return None


import xml.etree.ElementTree as ET
import os


def load_all_envs(RIxml_address='data/RouteInstructions/Route_Instructions_LongestShortestV8.xml'):
    envs = []
    tree = ET.parse(RIxml_address)
    root = tree.getroot()
    for letter in root.findall('letter'):
        lettertext = letter.get('name')
        for route in letter.findall('route'):

            # return *.z8 files in the directory 'data/Environments' if the beginning of the file name is the same as the letter text
            for file in os.listdir('data/Environments'):
                if file.startswith(lettertext) and file.endswith('.z8'):
                    envs.append({'lettertext': lettertext,
                                        'x_origin': route.get('x_origin'),
                                        'y_origin': route.get('y_origin'),
                                        'x_destination': route.get('x_destination'),
                                        'y_destination': route.get('y_destination'),
                                        'env': file})
    return envs

def learn_envs(Environment):
    env = TextWorldEnv('data/Environments/'+Environment['env'], Environment['x_destination'], Environment['y_destination'])
    env = Monitor(env, filename='data/Logs/monitor.log', allow_early_resets=True)

    reward_threshold = 495

    callbackOnBest = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    callbackOnNoImprovement = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=10, verbose=1)
    callback = EvalCallback(eval_env=env, best_model_save_path='data/EnvironmentName/',
                            log_path='data/Logs/EnvironmentName', eval_freq=20000, deterministic=False, render=False,
                            callback_after_eval=callbackOnNoImprovement, callback_on_new_best=callbackOnBest)

    model = PPO(policy="MlpPolicy", env=
    env, verbose=1, seed=0, device='cuda')
    model.learn(total_timesteps=500000, log_interval=1, callback=callback, tb_log_name='PPO')

    return model


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    all_envs = load_all_envs()
    print(len(all_envs))

    all_envs = all_envs[1:40]
    print(all_envs)
    print(torch.cuda.is_available())

    for Environment in all_envs:
        model = learn_envs(Environment)


















