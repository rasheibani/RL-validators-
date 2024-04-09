Environment = 'data/Environments/A_Average-Regular_Approach1_545604.9376088154_1000842.9379071898.z8'

import textworld
import gym
from gym import spaces
import torch
import numpy as np
import spacy
import re

def extract_coordinates(game_state):
    # Regular expression pattern to match the X and Y values
    pattern = r"X:\s*([\d.]+)\s*\nY:\s*([\d.]+)"

    matches = re.search(pattern, game_state)
    if matches:
        x = float(matches.group(1))
        y = float(matches.group(2))
        return np.array([x]), np.array([y])
    else:

        return np.array([0]), np.array([0])
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
            'text': spaces.Box(low=-np.inf, high=np.inf, shape=(300,), dtype='float32'),
            'x': spaces.Box(low=0, high=10000000, shape=(1,), dtype='float32'),
            'y': spaces.Box(low=0, high=10000000, shape=(1,), dtype='float32')})

    def reset(self):
        self.game_state = self.env.reset()
        self.x, self.y = extract_coordinates(self.game_state.feedback)
        feedback_embeding = feedback_to_embedding(self.game_state.feedback)
        return {'text': feedback_embeding.reshape(1,-1), 'x': self.x, 'y': self.y}

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
        if self.x == 545604.9376088154 and self.y == 1000842.9379071898:
            reward = 100
            done = True
        else:
            reward = 0
            done = False
        return {'text': feedback_embeding.reshape(1,-1), 'x': self.x, 'y': self.y}, reward, done, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __len__(self):
        return 1 # only one game running at a time


# # test the environment
# env = TextWorldEnv(Environment)
# obs = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, _ = env.step(action)
#     print(action, reward, done)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Agent
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state))
            target_f = self.model(state)
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training loop
def train_agent(env, agent, episodes, batch_size):
    for e in range(episodes):
        state = env.reset()
        state = np.array([state['x'][0], state['y'][0]])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array([next_state['x'][0], next_state['y'][0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # print(f"Episode: {e+1}/{episodes}")

# Set up the environment and agent
env = TextWorldEnv(Environment)
state_size = 2  # x and y coordinates
action_size = env.action_space.n
agent = Agent(state_size, action_size)

# Start training
train_agent(env, agent, episodes=200, batch_size=32)

# Save the trained model
torch.save(agent.model.state_dict(), 'model.pth')
# print learned tabular Q values (state, action) pairs for the environment

