# Environment = 'data/Environments/A_Average-Regular_Approach1_545604.9376088154_1000842.9379071898.z8'
import pandas

Environment = 'data/Environments/test.zblorb'
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
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, \
    StopTrainingOnNoModelImprovement

from stable_baselines3.common.monitor import Monitor
import multiprocessing
import Pretraining
from stable_baselines3.common.policies import obs_as_tensor
import xml.etree.ElementTree as ET
import os


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=64)
        n_actions = 8
        n_instructions = 15
        n_total_features = n_actions + n_instructions + 1

        self.observation_flat_dim = n_total_features

        # Define the feature extractor network
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.observation_flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # flatten the observations
        observations_flat = observations.view(observations.size(0), self.observation_flat_dim)

        return self.feature_extractor(observations_flat)


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule,
                                           features_extractor_class=CustomFeatureExtractor, *args, **kwargs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        admissible_actions = obs[:, :8]  # Extract the admissible actions
        route_instructions = obs[:, 8:23]  # Extract the route instructions
        instruction_index = obs[:, 23]  # Extract the instruction index

        valid_actions_mask = admissible_actions > 0
        features = self.extract_features(obs)
        distribution = self._get_action_dist_from_latent(features)
        admissible_actions_prob = distribution.log_prob(admissible_actions)

        # convert the probablity of actions that are not admissible to -inf
        admissible_actions_prob[~valid_actions_mask] = float('-inf')

        if deterministic:
            action = torch.argmax(admissible_actions_prob, dim=1)
        else:
            # sample from admissible actions where the probability is not -inf
            action = torch.multinomial(admissible_actions_prob.exp(), 1)

        # get the probability of the selected action
        prob_of_selected_action = admissible_actions_prob.gather(1, action)

        values = self.value_net(features)
        return action, values, prob_of_selected_action


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


def text_to_action(text):
    mapping = {
        'go north': 0,
        'go south': 1,
        'go east': 2,
        'go west': 3,
        'go northeast': 4,
        'go northwest': 5,
        'go southeast': 6,
        'go southwest': 7
    }
    return mapping.get(text, -1)


def normalize(observation):
    # Separate the components
    admissible_actions = observation[:8]  # Assuming first 8 are admissible actions
    route_instructions = observation[8:23]  # Next 15 are route instructions
    instruction_index = observation[23]  # Last is instruction index

    # Normalize admissible actions (already in [0, 1])
    normalized_admissible_actions = admissible_actions

    # Normalize route instructions, treating 8 as padding and replacing it with -1
    normalized_route_instructions = np.where(route_instructions != 8, route_instructions / 7, -1)

    # Normalize instruction index
    max_instruction_index = len(route_instructions)
    normalized_instruction_index = instruction_index / max_instruction_index

    # Combine normalized components
    normalized_observation = np.concatenate(
        [normalized_admissible_actions, normalized_route_instructions, [normalized_instruction_index]])

    return normalized_observation


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
        # Return a tensor with a valid shape filled with zeros
        return np.zeros((1, 300), dtype=np.float32)


def sentence_from_action(action):
    if action == 0:
        return "go north"
    elif action == 1:
        return "go south"
    elif action == 2:
        return "go east"
    elif action == 3:
        return "go west"
    elif action == 4:
        return "go northeast"
    elif action == 5:
        return "go northwest"
    elif action == 6:
        return "go southeast"
    elif action == 7:
        return "go southwest"
    else:
        return "look"


def extract_area_id(feedback):
    pattern = r"An area \((\d+)\) in r(\d+)"
    matches = re.search(pattern, feedback)
    if matches:
        area_id = matches.group(1)
        room_id = matches.group(2)
        return f"a{area_id}r{room_id}"
    else:
        return None


def admissible_actions_to_observation(admissible_actions):
    observation = np.zeros(8, dtype=np.int32)
    for i, direction in enumerate(
            ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']):
        if f"go {direction}" in admissible_actions:
            observation[i] = 1
    return observation


class TextWorldEnv(gym.Env):
    def __init__(self, game_address, x_destination, y_destination, n_instructions=1):
        self.y = None
        self.x = None
        self.n_instructions = n_instructions
        self.x_origin = None
        self.y_origin = None
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

        self.dist_from_origin_to_destination = None

        self.letsprint = True
        self.exploration_threshold = 0

        # The observation space is vector with 8 elements binary, each element representing a direction, 1 if the
        # direction is admissible, 0 otherwise
        self.observation_space = spaces.Box(low=0, high=8, shape=(24,), dtype=np.int32)
        self.route_instructions = []
        self.instruction_index = 0

    def reset(self, **kwargs):
        self.game_state = self.env.reset()
        self.game_state, _, _ = self.env.step("look")
        admissible_actions = get_admissible_actions(self.game_state.feedback)
        observation = admissible_actions_to_observation(admissible_actions)
        self.x, self.y = extract_coordinates(self.game_state.feedback)
        self.x_origin = self.x
        self.y_origin = self.y
        if 'route_instructions' in kwargs:
            rti = kwargs['route_instructions']
            self.x_destination, self.y_destination = self.get_destination_from_route_instructions(
                self.route_instructions)
            self.route_instructions = [text_to_action(instruction) for instruction in rti.split('. ')]
        else:
            while True:
                admissible_actions = get_admissible_actions(self.game_state.feedback)
                observation = admissible_actions_to_observation(admissible_actions)
                self.route_instructions, self.x_destination, self.y_destination = self.generate_route_instructions(
                    **kwargs)
                if len(self.route_instructions) > 0:
                    break
        self.dist_from_origin_to_destination = np.sqrt((self.x_destination - self.x_origin) ** 2 + (
                self.y_destination - self.y_origin) ** 2)
        self.visited_states_actions.clear()
        self.instruction_index = 0
        if len(self.route_instructions) < 15:
            route_instruction_padded = np.pad(self.route_instructions,
                                              (0, 15 - len(self.route_instructions)), 'constant', constant_values=8)
        else:
            route_instruction_padded = self.route_instructions[:15]
        observation = np.concatenate((
            admissible_actions_to_observation(admissible_actions),
            route_instruction_padded,
            np.array([self.instruction_index])
        ))
        observation = normalize(observation)
        return observation, {}

    def step(self, action):
        # if there is this sentence "You can't go that way."in the feedback, then execute a look command
        if "You can't go that way." in self.game_state.feedback:
            self.game_state, _, _ = self.env.step("look")
        # check if the action is within the admissible actions
        admissible_actions = get_admissible_actions(self.game_state.feedback)
        sentence = sentence_from_action(action)

        if sentence not in admissible_actions:
            # print(f"Action {sentence} is not admissible. Skipping.")
            reward = -1
            terminate = False
            truncated = False
            observation = np.concatenate((
                admissible_actions_to_observation(admissible_actions),
                np.pad(self.route_instructions,
                       (0, 15 - len(self.route_instructions)), 'constant', constant_values=8),
                np.array([self.instruction_index])
            ))
            observation = normalize(observation)
            print(''f'')

            return observation, reward, terminate, truncated, {}
        else:
            self.game_state, _, done_dummy = self.env.step(sentence)
            self.x, self.y = extract_coordinates(self.game_state.feedback)

            feedback_embedding = feedback_to_embedding(self.game_state.feedback)
            self.last_feedback_embedding = feedback_embedding
            admissible_action = get_admissible_actions(self.game_state.feedback)

            target_x = np.float64(self.x_destination)
            target_y = np.float64(self.y_destination)

            if np.isclose(self.x, target_x, atol=1e-3) and np.isclose(self.y, target_y, atol=1e-3):

                reward = 20
                terminate = True
                self.counter = self.counter + 1
                # print with green color
                # print(f'\033[92m{self.instruction_index}\033[0m')
            else:
                if self.instruction_index >= len(self.route_instructions) + self.exploration_threshold - 1:
                    distance = np.sqrt((self.x - self.x_destination) ** 2 + (self.y - self.y_destination) ** 2)
                    reward = -1
                    terminate = False
                    truncated = True
                    observation = np.concatenate((
                        admissible_actions_to_observation(admissible_actions),
                        np.pad(self.route_instructions,
                               (0, 15 - len(self.route_instructions)), 'constant', constant_values=8),
                        np.array([self.instruction_index])
                    ))
                    observation = normalize(observation)
                    # print(observation, 'truncated') if self.letsprint else None
                    self.instruction_index = self.instruction_index + 1
                    return observation, reward, terminate, truncated, {}
                # shape reward based on the distance to the target
                distance = np.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)
                reward = 0
                # reward = np.clip(reward, 0, 20)
                terminate = False
            truncated = False
            observation = np.concatenate((
                admissible_actions_to_observation(admissible_action),
                np.pad(self.route_instructions,
                       (0, 15 - len(self.route_instructions)), 'constant', constant_values=8),
                np.array([self.instruction_index])
            ))
            observation = normalize(observation)
            self.instruction_index = self.instruction_index + 1
            return observation, reward, terminate, truncated, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def __len__(self):
        return 1  # only one game running at a time

    def generate_route_instructions(self, **kwargs):

        # generate a random number form 1 to 15
        n_instructions = self.n_instructions
        # print(f'Number of Instructions: {n_instructions}')
        # run the env for n_instructions steps and collect the route instructions
        # make sure that route instructions are from the admissible actions
        route_instructions = []
        for i in range(n_instructions):
            admissible_actions = get_admissible_actions(self.game_state.feedback)
            # print(f'Admissible Actions: {admissible_actions}')
            instruction = np.random.choice(admissible_actions)
            # print(f'Instruction: {instruction}')
            route_instructions.append(instruction)
            self.game_state, _, _ = self.env.step(instruction)

        # convert the elements of the route instructions to integers
        route_instructions = [text_to_action(instruction) for instruction in route_instructions]

        # collect the destination x y
        self.x_destination, self.y_destination = extract_coordinates(self.game_state.feedback)
        # rest the env
        self.game_state = self.env.reset()
        self.game_state, _, _ = self.env.step("look")
        return route_instructions, self.x_destination, self.y_destination

    def get_destination_from_route_instructions(self, route_instructions):
        for instruction in route_instructions:
            self.game_state, _, _ = self.env.step(instruction)
        return extract_coordinates(self.game_state.feedback)


def learn_envs(environments):
    model = None
    for i, Environment in enumerate(environments):
        env_name = Environment['env']
        env_dir = f'data/trained/{env_name}'
        env_logs_dir = f'{env_dir}/Logs'
        env_model_dir = f'{env_dir}/Models'
        print(f"Training on {env_name}")

        n_instructions = 1

        if env_name == 'simplest_simplest_546025.6070834016_996382.4069940181.z8':
            n_instructions = 1
            # check if the env name start with one of the elements in pretraining set
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining25):
            n_instructions = 2
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining50):
            n_instructions = 4
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining75):
            n_instructions = 7
        elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining100):
            n_instructions = 10

        # Create and wrap the environment
        env = TextWorldEnv(f'data/Environments/{env_name}',
                           Environment['x_destination'], Environment['y_destination'], n_instructions=n_instructions)
        env = Monitor(env, filename=f'{env_logs_dir}/monitor.log', allow_early_resets=True)

        reward_threshold = 19

        callbackOnBest = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        callbackOnNoImprovement = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=10, verbose=1)
        callback = EvalCallback(
            eval_env=env,
            best_model_save_path=env_model_dir,
            log_path=env_logs_dir,
            eval_freq=500000,
            deterministic=False,
            render=False,
            # callback_after_eval=callbackOnNoImprovement,
            callback_on_new_best=callbackOnBest
        )

        # Load model if it exists, otherwise initialize it
        if model is None:
            model = DQN('MlpPolicy', env=env, verbose=20, seed=0, device='cuda', exploration_fraction=0.99)
        else:
            model.set_env(env)

        n_instructions = i + 1

        # Learn the model
        model.learn(total_timesteps=5000000, log_interval=25000, tb_log_name=f'PPO_{env_name}',
                    reset_num_timesteps=True)

        # Save the model after training
        model.save(f'{env_model_dir}/final_model')
        # copy the z8 file to the trained folder
        os.system(f'cp data/Environments/{env_name} {env_dir}/{env_name}')

    return model


def evaluate_model(model, environments):
    for i, Environment in enumerate(environments):
        env_name = Environment['env']
        env_dir = f'data/{env_name}'
        env_logs_dir = f'{env_dir}/Logs'
        env_model_dir = f'{env_dir}/Models'

        # Create and wrap the environment
        env = TextWorldEnv(f'data/Environments/{env_name}', Environment['x_destination'], Environment['y_destination'])
        env = Monitor(env, filename=f'{env_logs_dir}/monitor.log', allow_early_resets=True)

        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=False, render=False,
                                                  callback=None, reward_threshold=None, return_episode_rewards=False)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    return


def eval_by_interaction(model, env, route_instruction):
    # use the route instruction sentence sequences to choose the consecutive actions with the environment and get the reward
    env_name = env['env']
    env_dir = f'data/{env_name}'
    env_logs_dir = f'{env_dir}/Logs'

    # Create and wrap the environment
    env = TextWorldEnv(f'data/Environments/{env_name}', env['x_destination'], env['y_destination'])
    env = Monitor(env, filename=f'{env_logs_dir}/monitor.log', allow_early_resets=True)

    # reset the environment
    observation, info = env.reset()
    episode_reward = 0

    # split the route instruction into sentences
    sentences = route_instruction.split('. ')
    for sentence in sentences:
        # action, _ = model.predict(observation, deterministic=False)
        action = text_to_action(sentence)
        observation, reward, terminate, truncated, _ = env.step(action)
        # extract the probability distribution of the actions
        b = predict_proba(model, observation)
        # print b with only 2 decimal places
        b = np.round(b, 3)
        print(f"Action: {sentence}, Probability Distribution: {b}")
        # extract the probability of the action
        prob = b[0][action]

        # add the probability of the action to the episode reward
        episode_reward += prob

        print(f"Terminate: {terminate}, Accumulated Probability: {round(episode_reward, 2)}")
        if terminate or truncated:
            print("Terminating the episode")
            break


def load_envs():
    # load list of environments from pretraining
    pretraining_set = Pretraining.Pretraining25
    pretraining_set.extend(Pretraining.Pretraining50)
    pretraining_set.extend(Pretraining.Pretraining75)
    pretraining_set.extend(Pretraining.Pretraining100)
    # search in data/Environment and see if any *.z8 files begin with pretraining_set element
    all_env_pretraining = []
    for env in pretraining_set:
        for file in os.listdir('data/Environments'):
            if file.startswith(env) and file.endswith('.z8'):
                env_name = file
                # split the name by _
                env_name = env_name.split('_')
                x_destination = float(env_name[-2])
                y_destination = float(env_name[-1].split('.')[0])
                all_env_pretraining.append({'env': file
                                               , 'x_destination': x_destination
                                               , 'y_destination': y_destination})
    print(all_env_pretraining)
    return all_env_pretraining


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


def evaluate_all_trained_models():
    # iterate on subfolders of data/trained and load the models

    df = pandas.DataFrame(columns=['Model', 'Mean Reward', 'Std Reward', 'Complexity_of_Environment'])
    for subfolder in os.listdir('data/trained'):
        model = DQN.load(f'data/trained/{subfolder}/Models/final_model.zip')
        env = TextWorldEnv(f'data/trained/{subfolder}/{subfolder}', 0, 0)
        # evaluate the model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False, render=False,
                                                  callback=None, reward_threshold=None, return_episode_rewards=True, warn=False)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        complexity = 0
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining25):
            complexity = 0.25
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining50):
            complexity = 0.5
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining75):
            complexity = 0.75
        if any(subfolder.startswith(envP) for envP in Pretraining.Pretraining100):
            complexity = 1
        if subfolder.startswith('simplest'):
            complexity = 0

        df = df._append(
            {'Model': subfolder.split('_')[0:2], 'Mean Reward': mean_reward, 'Std Reward': std_reward,
             'Complexity_of_Environment': complexity}, ignore_index=True)
        df.sort_values(by=['Complexity_of_Environment'], inplace=True)
        df.to_csv('data/evaluation_results.csv')
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        print(f"Model {subfolder} evaluated successfully")


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    print(torch.cuda.is_available())

    # load the environments
    all_env_pretraining = load_envs()

    # learn the environments in all_env_pretraining
    model = learn_envs(all_env_pretraining)
    evaluate_all_trained_models()


    # # evaluate the model
    # model = PPO.load('data/trained/simplest_simplest_546025.6070834016_996382.4069940181.z8/Models/final_model.zip')
    # env = TextWorldEnv('data/Environments/simplest_simplest_546025.6070834016_996382.4069940181.z8', 996382.4069940181,
    #                    996382.4069940181)
    # # change the seed of np random generator
    # np.random.seed(0)
    # # evaluate_model(model, all_envs)
    #
    # route_instruction = 'go east. go south. go west. go southwest. go southwest. Arrive at destination!'
    # another_route_instruction = 'go west'
    # observation, _ = env.reset(route_instructions=another_route_instruction)
    # # eval_by_interaction(model, all_envs[0], route_instruction)
    # print(f'observation: {observation}')
    # b = predict_proba(model, observation)

