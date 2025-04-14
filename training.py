import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from z8file_to_dictionaries import z8file_to_dictionaries
from environment import TextWorldEnv
from utils import determine_complexity, determine_n_instructions

class CustomStopOnNoImprovement(BaseCallback):
    def __init__(self, max_no_improvement_evals: int = 3, min_evals: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.eval_count = 0
        self.no_improvement_count = 0
        self.best_reward = -np.inf

    def _on_training_start(self) -> None:
        self.eval_count = 0
        self.no_improvement_count = 0
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.eval_count >= self.min_evals:
            eval_reward = self.evaluate_model()
            self.eval_count += 1
            if eval_reward > self.best_reward:
                self.best_reward = eval_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            if self.no_improvement_count >= self.max_no_improvement_evals:
                self.logger.info("Stopping training due to no improvement.")
                self.model.stop_training = True

    def evaluate_model(self) -> float:
        total_reward = 0
        eval_episodes = 5
        for _ in range(eval_episodes):
            obs = self.training_env.reset()
            done = False
            episode_reward = 0
            timestep = 0
            while not done and timestep < 100:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.training_env.step(action)
                episode_reward += reward
                timestep += 1
            if timestep >= 100:
                done = True
            total_reward += episode_reward
        avg_reward = total_reward / eval_episodes
        self.logger.info(f"Evaluation reward: {avg_reward}")
        return avg_reward

    def _on_training_end(self) -> None:
        pass

def learn_envs(environments, max_iterations=10000):
    models = {}
    reward_types = ['sparse', 'step_cost']
    sorted_envs = sorted(environments, key=lambda x: determine_complexity(x['env']))

    for reward_type in reward_types:
        print(f"\n=== Training with reward type: {reward_type} ===")
        model = None
        curriculum_model_dir = f'data/trained/curriculum/PPO/{reward_type}'
        os.makedirs(curriculum_model_dir, exist_ok=True)
        model_path = os.path.join(curriculum_model_dir, 'best_model.zip')

        for env_info in sorted_envs:
            env_name = env_info['env']
            print(f"\nTraining on environment: {env_name}")
            gameaddress = f'data/Environments/{env_name}'
            try:
                game_dict, room_positions = z8file_to_dictionaries(gameaddress)
            except Exception as e:
                print(f"Skipping {env_name} due to error: {e}")
                continue

            env = TextWorldEnv(
                game_dict=game_dict,
                room_positions=room_positions,
                n_instructions=determine_n_instructions(env_name),
                grammar=8,
                reward_type=reward_type
            )
            env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
            env = Monitor(env, filename=f'{curriculum_model_dir}/monitor_{env_name}.log')

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

            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=24.9, verbose=1)
            eval_callback = EvalCallback(
                env,
                best_model_save_path=curriculum_model_dir,
                eval_freq=5000,
                callback_on_new_best=stop_callback,
                verbose=0
            )

            model.learn(
                total_timesteps=max_iterations,
                callback=eval_callback,
                tb_log_name=f"PPO_curriculum_{reward_type}",
                reset_num_timesteps=False
            )

            model.save(model_path)
            print(f"Saved trained model to {model_path}")

        models[reward_type] = model
    return models
