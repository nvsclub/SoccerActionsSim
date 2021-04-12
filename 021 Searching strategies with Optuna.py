from env.SoccerActionsEnv import SoccerActionsEnv

import pandas as pd
import numpy as np

import lib.draw as draw
import matplotlib.pyplot as plt
from tqdm import tqdm

import optuna

def calculate_horizontal_position(x, y):
    if x < 0.75:
        if y < 0.25:
            return 0
        elif y < 0.5:
            return 1
        elif y < 0.75:
            return 2
        else:
            return 3
    else:
        if y < 0.2037:
            return 0
        elif y < 0.3653:
            return 1
        elif y < 0.50:
            return 2
        elif y < 0.6347:
            return 3
        elif y < 0.7963:
            return 4
        else:
            return 5

def calculate_square(x, y):
    if x < 0.1666:
        return 0 + calculate_horizontal_position(x, y)
    elif x < 0.3333:
        return 4 + calculate_horizontal_position(x, y)
    elif x < 0.5:
        return 8 + calculate_horizontal_position(x, y)
    elif x < 0.6666:
        return 12 + calculate_horizontal_position(x, y)
    elif x < 0.75:
        return 16 + calculate_horizontal_position(x, y)
    elif x < 0.8428:
        return 20 + calculate_horizontal_position(x, y)
    elif x < 0.9476:
        return 26 + calculate_horizontal_position(x, y)
    else:
        return 32 + calculate_horizontal_position(x, y)

def test_model(action, r, a):
    env = SoccerActionsEnv(randomized_start=True, end_on_xg=True)
    obs = env.reset()

    saving_rewards = []
    for i in tqdm(range(10000)):
        pos = calculate_square(obs[0], obs[1])
        obs, rewards, done, info = env.step([action[pos], r[pos], a[pos]])
        if done:
            saving_rewards.append(info['expectedGoals'])
            env.reset()

    return np.mean(saving_rewards)


# Optimization function
def objective(trial):
    action = []
    r = []
    a = []
    for i in range(38):
        action.append(trial.suggest_uniform('act'+str(i), 0, 1))
        r.append(trial.suggest_uniform('r'+str(i), 0, 1))
        a.append(trial.suggest_uniform('a'+str(i), 0, 1))
    
    return test_model(action, r, a)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5000)
study.trials_dataframe().to_csv('saved_models/optuna_optization_process.csv')

xxs = [8.33, 8.33, 8.33, 8.33, 25, 25, 25, 25, 41.66, 41.66, 41.66, 41.66, 58.33, 58.33, 58.33, 58.33, 70.83, 70.83, 70.83, 70.83, 79.64, 79.64, 79.64, 79.64, 79.64, 79.64, 89.52, 89.52, 89.52, 89.52, 89.52, 89.52, 97.38, 97.38, 97.38, 97.38, 97.38, 97.38]
yys = [12.5, 37.5, 62.5, 87.5, 12.5, 37.5, 62.5, 87.5, 12.5, 37.5, 62.5, 87.5, 12.5, 37.5, 62.5, 87.5, 12.5, 37.5, 62.5, 87.5, 10.19, 28.45, 43.27, 56.73, 71.55, 89.81, 10.19, 28.45, 43.27, 56.73, 71.55, 89.81, 10.19, 28.45, 43.27, 56.73, 71.55, 89.81]

df = pd.DataFrame(xxs, columns=['x'])
df['y'] = yys
df['i'] = [i for i in range(38)]
df['action'] = [study.best_params['act'+str(i)] for i in range(38)]
df['r'] = [study.best_params['r'+str(i)] for i in range(38)]
df['a'] = [study.best_params['a'+str(i)] for i in range(38)]

df.to_csv('saved_models/optuna_results.csv')