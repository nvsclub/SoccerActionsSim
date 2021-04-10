import gym
from gym import spaces
import pickle
from random import random
import numpy as np


class SoccerActionsEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, permanent_x=50, permanent_y=50, randomized_start=True, end_on_xg=True, deterministic=True):
        super(SoccerActionsEnv, self).__init__()

        # Loading models
        self.pass_model = pickle.load(open('env/matrix/pass_gradient.sav', 'rb'))
        self.shot_model = pickle.load(open('env/matrix/shot_gradient.sav', 'rb'))

        # Saving parameters
        self.permanent_x = permanent_x
        self.permanent_y = permanent_y
        self.randomized_start = randomized_start
        self.end_on_xg = end_on_xg
        self.deterministic = deterministic

        # Define action and observation space
        self.action_space = spaces.Box(low=np.float32(np.array([0, 0, 0])), high=np.float32(np.array([1, 1, 1])))
        self.observation_space = spaces.Box(low=np.float32(np.array([0, 0])), high=np.float32(np.array([1, 1])))

        # Action storage
        self.action_storage = []

    def toggle_deterministic_env(self):
        self.deterministic = True
    
    def toggle_probabilistic_env(self):
        self.deterministic = False

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        original_state = self._next_observation()
        
        # Action[0] defines if action is a shot or a pass
        if self.deterministic:
            if action[0] < 0.5:
                self.xg = self.shot_model.predict_proba([[self.x, self.y]])[0,1]
                self.goal = random() < self.xg
                self.done = True

                if self.end_on_xg:
                    self.reward += self.xg
                else:
                    self.reward += self.goal
            else:
                # Action[1,2] are the pointers for the pass action
                outcome_probability = self.pass_model.predict_proba([[self.x, self.y, action[1], action[2]]])[0,1]
                self.done = random() > outcome_probability
                
                if not self.done:
                    prev_x = self.x
                    self.x = self.x + action[1] * np.cos((action[2] - 0.5) * 2 * np.pi)
                    self.y = self.y + action[1] * np.sin((action[2] - 0.5) * 2 * np.pi)

                    if not (0 < self.y < 1) and not (0 < self.x < 1):
                        self.done = True

                    # Reward sucessful passes, but not if the pass was backwards
                    self.reward += 0.0001 * (self.x > prev_x)
        # Action[0] defines the probabilities of the doing action
        else:
            # Pass
            if random() < action[0]:
                # Action[1,2] are the pointers for the pass action
                outcome_probability = self.pass_model.predict_proba([[self.x, self.y, action[1], action[2]]])[0,1]
                self.done = random() > outcome_probability
                
                if not self.done:
                    prev_x = self.x
                    self.x = self.x + action[1] * np.cos((action[2] - 0.5) * 2 * np.pi)
                    self.y = self.y + action[1] * np.sin((action[2] - 0.5) * 2 * np.pi)

                    if not (0 < self.y < 1) and not (0 < self.x < 1):
                        self.done = True

                    # Reward sucessful passes, but not if the pass was backwards
                    self.reward += 0.0001 * (self.x > prev_x)
            # Shot
            else:
                self.xg = self.shot_model.predict_proba([[self.x, self.y]])[0,1]
                self.goal = random() < self.xg
                self.done = True

                if self.end_on_xg:
                    self.reward += self.xg
                else:
                    self.reward += self.goal


        obs = self._next_observation()

        # Save information for visualization
        self.action_storage.append({'action': action, 'observation': original_state, 'success': self.goal or not self.done})

        return obs, self.reward, self.done, {'expectedGoals': self.xg}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.reward = 0
        self.xg = 0
        self.goal = False
        self.done = False
        self.current_step = 0
        self.action_storage = []

        if self.randomized_start:
            self.permanent_x = random()
            self.permanent_y = random()

        # State Space
        self.x = self.permanent_x
        self.y = self.permanent_y
        
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([self.x, self.y])
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if self.done:
            print(f'Steps: {self.current_step} - Reward: {self.reward}')