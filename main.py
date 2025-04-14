import spacy
import torch
import warnings
from utils import load_envs
from training import learn_envs
from evaluation import evaluate_curriculum_models
import numpy as np


def predict_proba(model, state):
    print(state)
    obs = model.policy.obs_to_tensor(state)[0]
    print(obs)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    print(probs)
    probs_np = probs.detach().cpu().numpy()
    probs_np = probs_np / np.sum(probs_np)
    return probs_np

def eval_by_interaction(model, env_info, route_instruction):
    from environment import TextWorldEnv, text_to_action
    from z8file_to_dictionaries import z8file_to_dictionaries
    from stable_baselines3.common.monitor import Monitor

    env_name = env_info['env']
    env_dir = f'data/{env_name}'
    env_logs_dir = f'{env_dir}/Logs'
    gameaddress = f'data/Environments/{env_name}'
    game_dict, room_positions = z8file_to_dictionaries(gameaddress)
    env = TextWorldEnv(
        game_dict=game_dict,
        room_positions=room_positions,
        x_destination=env_info['x_destination'],
        y_destination=env_info['y_destination'],
        grammar=8,
        reward_type='sparse'
    )
    env = Monitor(env, filename=f'{env_logs_dir}/monitor.log', allow_early_resets=True)
    observation, info = env.reset()
    episode_reward = 0
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
    nlp = spacy.load("en_core_web_sm")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    warnings.filterwarnings("ignore", category=UserWarning)
    all_env_pretraining = load_envs()
    learn_envs(all_env_pretraining, max_iterations=50000)
    evaluate_curriculum_models(max_seen_envs=2, max_unseen_envs=2, random_seed=2)
