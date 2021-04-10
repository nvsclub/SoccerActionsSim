import tensorflow

import gym
import json
import datetime as dt

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from env.SoccerActionsEnv import SoccerActionsEnv

import pandas as pd
import numpy as np

import lib.draw as draw
import matplotlib.pyplot as plt
from tqdm import tqdm

import optuna

env = SoccerActionsEnv(randomized_start=True, end_on_xg=True)

def test_model(env, model, name):
    saving_rewards = []
    obs = env.reset()
    for i in tqdm(range(20000)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            saving_rewards.append(rewards)
            env.reset()

    return np.mean(saving_rewards)


# Optimization function
def objective(trial):
    noise = trial.suggest_uniform('Noise', 0.1, 0.8)
    timesteps = trial.suggest_int('Timesteps', 10, 100)

    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(noise) * np.ones(n_actions))
    model = DDPG('MlpPolicy', env, action_noise=action_noise)
    model.learn(total_timesteps=timesteps*1000, log_interval=1000)
    
    return test_model(env, model, '')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)
study.trials_dataframe().to_csv('saved_models/optuna_optimization_noise-duration.csv')
