import tensorflow

import gym
import json
import datetime as dt

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from env.SoccerActionsEnv import SoccerActionsEnv

import pandas as pd
import numpy as np

import lib.draw as draw
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def make_env(rank, seed=0):
    def _init():
        env = SoccerActionsEnv(randomized_start=True, end_on_xg=True)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env = SubprocVecEnv([make_env(i) for i in range(1)])

    n_actions = env.action_space.shape[-1]

    t1 = time.time()

    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=25000, log_interval=10000)
    print(time.time() - t1)

    saving_rewards = []
    obs = env.reset()
    for i in tqdm(range(10000)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()
        if done:
            saving_rewards.append(rewards)
            env.reset()
    np.mean(saving_rewards)    