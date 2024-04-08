Environment = 'data/Environments/A_Average-Regular_Approach1_545604.9376088154_1000842.9379071898.z8'

from QLearning import extract_coordinates
import textworld
import gym
from gym import spaces
import torch
import numpy as np

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
            'text': spaces.Box(low=0, high=255, shape=(1,), dtype='uint8'),
            'x': spaces.Box(low=0, high=1000000, shape=(1,), dtype='float32'),
            'y': spaces.Box(low=0, high=1000000, shape=(1,), dtype='float32')})

    def reset(self):
        self.game_state = self.env.reset()
        self.x, self.y = extract_coordinates(self.game_state.feedback)
        return {'text': np.array([ord(c) for c in self.game_state.feedback]), 'x': self.x, 'y': self.y}, {}

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
        if self.x == 545604.9376088154 and self.y == 1000842.9379071898:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
        return {'text': np.array([ord(c) for c in self.game_state.feedback]), 'x': self.x, 'y': self.y}, reward, done, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __len__(self):
        return 1 # only one game running at a time


# test the environment
env = TextWorldEnv(Environment)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(action, reward, done)

# Now we can use this environment to train a reinforcement learning agent
# to find the destination in the textworld game
# we can use Q-learning algorithm to train the agent

import numpy as np
import tianshou
from tianshou.utils.net.common import Net
from tianshou.policy import DQNPolicy
from torch.utils.tensorboard import SummaryWriter


net = Net(state_shape=env.observation_space['text'].shape, action_shape=env.action_space.n)

# define the policy
policy = DQNPolicy(net, optim=torch.optim.Adam(net.parameters(), lr=1e-3), discount_factor=0.9)

# define the collector
train_collector = tianshou.data.Collector(policy, env, buffer=None)
test_collector = tianshou.data.Collector(policy, env, buffer=None)

# train the agent
result = tianshou.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10,
    step_per_epoch=1000,
    step_per_collect=10,
    update_per_step=0.1,
    episode_per_test=100,
    batch_size=64,
    train_fn= lambda epoch, env_step: policy.set_eps(0.1),
    test_fn= lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn= lambda mean_rewards: mean_rewards >= 95,
    logger=SummaryWriter('log.txt')
)

# print the result
print(f'Finished training! Use {result["duration"]}')
print(f'Final reward: {result["best_reward"]}')

