import random
import os
import pandas as pd
import numpy as np
import Pretraining


def determine_complexity(env_name):
    with open('data/for_complexity.csv', 'r') as f:
        df = pd.read_csv(f)
        for i in range(len(df)):
            if env_name.startswith(df['distinct'][i]):
                return df['Normalized Bearing Entropy'][i]

        if env_name.startswith('simplest'):
            return 0
        elif len(env_name.split('_')[0]) <= 1:
            return 0
        elif len(env_name.split('_')[0]) == 2:
            return 0.27
        elif len(env_name.split('_')[0]) >= 3:
            return 0.52
        return 0.77
    return 1000


def assign_complexity_category_updated(complexity):
    if complexity < 0.26:
        return '0-0.26'
    elif 0.26 <= complexity < 0.51:
        return '0.26-0.51'
    elif 0.51 <= complexity < 0.76:
        return '0.51-0.76'
    elif 0.76 <= complexity <= 1:
        return '0.76-1'
    return 'Unknown'


def load_envs():
    pretraining_set = Pretraining.Pretraining25 + Pretraining.Pretraining50 + \
                      Pretraining.Pretraining75 + Pretraining.Pretraining100
    all_env_pretraining = []
    for env_prefix in pretraining_set:
        for file in os.listdir('data/Environments'):
            if file.startswith(env_prefix) and file.endswith('.z8'):
                env_name = file
                env_name_parts = env_name.split('_')
                try:
                    x_destination = float(env_name_parts[-2])
                    y_destination = float(env_name_parts[-1].split('.')[0])
                except (IndexError, ValueError):
                    x_destination = 0.0
                    y_destination = 0.0
                all_env_pretraining.append({
                    'env': file,
                    'x_destination': x_destination,
                    'y_destination': y_destination
                })
    return all_env_pretraining


def get_all_possible_envs():
    all_envs = []
    for file in os.listdir('data/Environments'):
        if file.endswith('.z8'):
            env_name = file
            env_name_parts = env_name.split('_')
            try:
                x_destination = float(env_name_parts[-2])
                y_destination = float(env_name_parts[-1].split('.')[0])
            except (IndexError, ValueError):
                x_destination = 0.0
                y_destination = 0.0
            n_instructions = determine_n_instructions(env_name)
            all_envs.append({
                'env': file,
                'x_destination': x_destination,
                'y_destination': y_destination,
                'n_instructions': n_instructions
            })
    return all_envs


def determine_n_instructions(env_name):
    if env_name.startswith('simplest'):
        n_instructions = 1
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining25):
        n_instructions = 2
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining50):
        n_instructions = 3
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining75):
        n_instructions = 4
    elif any(env_name.startswith(envP) for envP in Pretraining.Pretraining100):
        n_instructions = 5
    else:
        n_instructions = 1
    return n_instructions
