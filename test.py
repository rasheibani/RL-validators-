Environment = 'data/Environments/A_Average-Regular_Approach1_545604.9376088154_1000842.9379071898.z8'

import textworld
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
import re
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
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
# Load the pre-trained English model
nlp = spacy.load("en_core_web_sm")


def feedback_to_embedding(feedback):
    doc = nlp(feedback)
    embeddings = []
    for token in doc:
        embeddings.append(token.vector)

    if embeddings:
        return np.mean(embeddings, axis=0).astype(np.float32)
    else:
        return np.zeros((300,), dtype=np.float32)

try:
    env = textworld.start(Environment)
except Exception as e:
    print(f"Failed to start game for letter {Environment}: {e}")

game_state = env.reset()
sentence1 = "Go north."
game_state, r, done = env.step(sentence1)
x, y = extract_coordinates(game_state.feedback)

if x == 545604.9376088154 and y == 1000842.9379071898:
    print("destination")
else:
    print("not destination")


# use gym to create the environment from the textworld game (it is partially observalbe)
# so we do not know all the states beforehand

class TextWorldEnv(gym.Env):
    def __init__(self, game_address):
        self.y = None
        self.x = None
        self.is_async=False
        self.game_state = None
        self.game_address = game_address
        self.env = textworld.start(game_address)
        # define this directions as action spaces : 'go left, go right, go forward, go back'
        self.action_space = spaces.Discrete(4)

        # The observation space is a Dict space with the following keys:
        # - 'text': a string representing the current game state
        # - 'x': the x-coordinate of the player
        # - 'y': the y-coordinate of the player
        self.observation_space = spaces.Dict({
            'text': spaces.Box(low=-np.inf, high=np.inf, shape=(96,), dtype='float32'),
            'x': spaces.Box(low=0, high=10000000, shape=(1,), dtype='float32'),
            'y': spaces.Box(low=0, high=10000000, shape=(1,), dtype='float32')})

    def reset(self, **kwargs):
        self.game_state = self.env.reset()
        self.x, self.y = extract_coordinates(self.game_state.feedback)
        feedback_embedding = feedback_to_embedding(self.game_state.feedback)
        observation = {'text': feedback_embedding.reshape(1, -1), 'x': self.x, 'y': self.y}
        info = {}  # Create an empty info dictionary
        return observation, info

    def step(self, action):
        if action == 0:
            sentence = "Go north."
        elif action == 1:
            sentence = "Go south."
        elif action == 2:
            sentence = "Go east."
        elif action == 3:
            sentence = "Go west."
        self.game_state, reward, done = self.env.step(sentence)
        self.x, self.y = extract_coordinates(self.game_state.feedback)
        feedback_embeding = feedback_to_embedding(self.game_state.feedback)
        target_x = 545604.9376088154
        target_y = 1000842.9379071898
        distance = np.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)

        if self.x == target_x and self.y == target_y:
            reward = 100
            done = True
        elif distance < 10:
            reward = 10
            done = False
        else:
            reward = -1
            done = False
        truncated = False
        # Add debugging lines

        return {'text': feedback_embeding.reshape(1, -1), 'x': self.x, 'y': self.y}, reward, done, truncated, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __len__(self):
        return 1 # only one game running at a time


env = TextWorldEnv(Environment)
model = DQN('MultiInputPolicy', env, verbose=1, learning_rate=0.1, gamma=0.9, tensorboard_log="./dqn_log/")
model.learn(total_timesteps=2000)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# print q-values
q_net = model.policy.q_net
print(q_net)

