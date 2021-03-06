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

n_tests = 10

env = SoccerActionsEnv(randomized_start=True, end_on_xg=True)

def test_model(env, model, name):
    saving_rewards = []
    saving_action_count = []
    obs = env.reset()
    count = 0
    for i in tqdm(range(10000)):
        count += 1
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            saving_rewards.append(info['expectedGoals'])
            saving_action_count.append(count)
            count = 0
            env.reset()
    print(name, np.mean(saving_rewards), np.mean(saving_action_count))

    f = open('saved_models/results.txt', 'a')
    f.write(f'{name},{np.mean(saving_rewards)},{np.mean(saving_action_count)}\n')
    f.close()

# A2C algorithm
for i in range(n_tests):
    test_name = 'saved_models/a2c_soccer_actions_env_1_' + str(i)
    n_actions = env.action_space.shape[-1]
    model = A2C('MlpPolicy', env)
    model.learn(total_timesteps=25000, log_interval=1000)
    model.save(test_name)
    test_model(env, model, test_name)

# DDPG algorithm
for i in range(n_tests):
    test_name = 'saved_models/ddpg_soccer_actions_env_1_' + str(i)
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))
    model = DDPG('MlpPolicy', env, action_noise=action_noise)
    model.learn(total_timesteps=10000, log_interval=1000)
    model.save(test_name)
    test_model(env, model, test_name)

for i in range(n_tests):
    test_name = 'saved_models/ddpg_soccer_actions_env_2_' + str(i)
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))
    policy_kwargs = dict(net_arch=[400, 300])
    model = DDPG('MlpPolicy', env, action_noise=action_noise, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=10000, log_interval=1000)
    model.save(test_name)
    test_model(env, model, test_name)

for i in range(n_tests):
    test_name = 'saved_models/ddpg_soccer_actions_env_3_' + str(i)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))
    model = DDPG('MlpPolicy', env, action_noise=action_noise)
    model.learn(total_timesteps=10000, log_interval=1000)
    model.save(test_name)
    test_model(env, model, test_name)

# PPO algorithm
for i in range(n_tests):
    test_name = 'saved_models/ppo_soccer_actions_env_1_' + str(i)
    n_actions = env.action_space.shape[-1]
    model = PPO('MlpPolicy', env)
    model.learn(total_timesteps=10000, log_interval=1000)
    model.save(test_name)
    test_model(env, model, test_name)

# SAC algorithm
for i in range(n_tests):
    test_name = 'saved_models/sac_soccer_actions_env_1_' + str(i)
    n_actions = env.action_space.shape[-1]
    model = SAC('MlpPolicy', env)
    model.learn(total_timesteps=10000, log_interval=1000)
    model.save(test_name)
    test_model(env, model, test_name)

# TD3 algorithm
for i in range(n_tests):
    test_name = 'saved_models/td3_soccer_actions_env_1_' + str(i)
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))
    model = TD3('MlpPolicy', env, action_noise=action_noise)
    model.learn(total_timesteps=10000, log_interval=1000)
    model.save(test_name)
    test_model(env, model, test_name)

for i in range(n_tests):
    test_name = 'saved_models/td3_soccer_actions_env_2_' + str(i)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))
    model = TD3('MlpPolicy', env, action_noise=action_noise)
    model.learn(total_timesteps=10000, log_interval=1000)
    model.save(test_name)
    test_model(env, model, test_name)


test_name = 'saved_models/ddpg_soccer_actions_env_2_BIG'
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))
policy_kwargs = dict(net_arch=[400, 300])
model = DDPG('MlpPolicy', env, action_noise=action_noise, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=100000, log_interval=1000)
model.save(test_name)
test_model(env, model, test_name)

test_name = 'saved_models/ddpg_soccer_actions_env_2_BIG1'
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))
policy_kwargs = dict(net_arch=[400, 300])
model = DDPG('MlpPolicy', env, action_noise=action_noise, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000000, log_interval=1000)
model.save(test_name)
test_model(env, model, test_name)

test_name = 'saved_models/ddpg_soccer_actions_env_2_BIG2'
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))
policy_kwargs = dict(net_arch=[400, 300])
model = DDPG('MlpPolicy', env, action_noise=action_noise, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000000, log_interval=1000)
model.save(test_name)
test_model(env, model, test_name)
