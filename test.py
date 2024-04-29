Environment = 'data/Environments/A_Average-Regular_Approach1_545604.9376088154_1000842.9379071898.z8'
RI = 'go east. go south. go west. go southwest. go southwest. Arrive at destination!'

import textworld
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


# try:
#     env = textworld.start(Environment)
# except Exception as e:
#     print(f"Failed to start game for letter {Environment}: {e}")
#
# game_state = env.reset()
# sentence1 = "Go north."
# game_state, r, done = env.step(sentence1)
# x, y = extract_coordinates(game_state.feedback)
#
# if x == 546281.0 and y == 1000842.9379071898:
#     print("destination")
# else:
#     print("not destination")


# use gym to create the environment from the textworld game (it is partially observalbe)
# so we do not know all the states beforehand

class TextWorldEnv(gym.Env):
    def __init__(self, game_address, total_steps=10000):
        self.y = None
        self.x = None
        self.is_async = False
        self.game_state = None
        self.game_address = game_address
        self.env = textworld.start(game_address)
        self.action_space = spaces.Discrete(8)
        self.visited_states_actions = set()
        self.last_feedback_embedding = None
        self.counter = 0

        # The observation space is a Dict space with the following keys:
        # - 'text': a string representing the current game state
        # - 'x': the x-coordinate of the player
        # - 'y': the y-coordinate of the player
        self.observation_space = spaces.Dict({
            'text': spaces.Box(low=-np.inf, high=np.inf, shape=(96,), dtype='float32'),
            'x': spaces.Box(low=544900, high=547300, shape=(1,), dtype='float32'),
            'y': spaces.Box(low=545000, high=546010, shape=(1,), dtype='float32')})

    def reset(self, **kwargs):
        self.game_state = self.env.reset()
        self.x, self.y = extract_coordinates(self.game_state.feedback)
        x_as_array = np.array([self.x], dtype='float32')
        y_as_array = np.array([self.y], dtype='float32')
        feedback_embedding = feedback_to_embedding(self.game_state.feedback)
        observation = {'text': feedback_embedding, 'x': x_as_array, 'y': y_as_array}
        info = {}  # Create an empty info dictionary
        self.visited_states_actions.clear()
        return observation, info

    def step(self, action):
        # check if the action is within the admissible actions
        admissible_actions = get_admissible_actions(self.game_state.feedback)
        # print(self.game_state.feedback)
        # print(admissible_actions)

        sentence = ""
        if action == 0:
            sentence = "go north"
        elif action == 1:
            sentence = "go south"
        elif action == 2:
            sentence = "go east"
        elif action == 3:
            sentence = "go west"
        elif action == 4:
            sentence = "go northeast"
        elif action == 5:
            sentence = "go northwest"
        elif action == 6:
            sentence = "go southeast"
        else:
            sentence = "go southwest"

        if sentence not in admissible_actions:
            # print(f"Action {sentence} is not admissible. Skipping.")
            reward = -0.07
            terminate = False
            truncated = False
            observation = {'text': self.last_feedback_embedding, 'x': np.array([self.x], dtype='float32'),
                           'y': np.array([self.y], dtype='float32')}
            return observation, reward, terminate, truncated, {}
        else:
            # print(sentence)
            self.game_state, reward, done_dummy = self.env.step(sentence)
            self.x, self.y = extract_coordinates(self.game_state.feedback)
            feedback_embedding = feedback_to_embedding(self.game_state.feedback)
            self.last_feedback_embedding = feedback_embedding
            target_x = 546281.0
            target_y = 999991.0
            distance = np.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)

            if np.isclose(self.x, target_x, atol=1e-3) and np.isclose(self.y, target_y, atol=1e-3):
                reward = 5000
                terminate = True
                # print(self.game_state.feedback)
                self.counter = self.counter + 1
                # print(self.counter)

                # print('----------------------------------------------------------------------')
                # print('---------Destination reached------------------------------------------')

            else:
                reward = 0
                terminate = False
            truncated = False
            self.visited_states_actions.add((self.x, self.y, action))
            # if self.counter > 0:
            #     print(f"X: {self.x}, Y: {self.y}, Action: {action}, Reward: {reward}")

            x_as_array = np.array([self.x], dtype='float32')
            y_as_array = np.array([self.y], dtype='float32')
            observation = {'text': feedback_embedding, 'x': x_as_array, 'y': y_as_array}


            return observation, reward, terminate, truncated, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __len__(self):
        return 1  # only one game running at a time


if __name__ == "__main__":
    counter = 0
    total_steps = 1e7
    # Load the pre-trained English model
    nlp = spacy.load("en_core_web_sm")
    env = TextWorldEnv(Environment)
    env2 = make_vec_env(TextWorldEnv, n_envs=16, env_kwargs={'game_address': Environment})

    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)
    # do the training withe high exploration rate

    model = DQN('MultiInputPolicy', env2, verbose=1,
                train_freq=1,
                exploration_final_eps=0.05,
                exploration_initial_eps=0.9,
                exploration_fraction=0.1,
                gradient_steps=4,
                gamma=0.8,
                learning_rate=0.0001,
                seed=0, stats_window_size=1)
    # make sure to include parameters of the trained model in the log file
    model.learn(total_timesteps=total_steps, log_interval=1, tb_log_name="./dqn_test")

    # model = DQN('MultiInputPolicy', env, verbose=1, train_freq=(1,"episode"),
    #             exploration_final_eps=0.65, exploration_initial_eps=0.9, gradient_steps=1, gamma=0.1, learning_rate=0.1)
    # model.learn(total_timesteps=total_steps, log_interval=1000, tb_log_name="./dqn_textworld")
    #
    # # Evaluate the agent
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    #
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    #
    model.save("dqn_textworld001")
    del model
    #
    # model = DQN.load("dqn_textworld5")

    # do the testing with PPO model
    # model = PPO('MultiInputPolicy', env, verbose=1)
    # model.learn(total_timesteps=total_steps, log_interval=1)
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")






