#################################################################################
# Provides a wrapper around gym environments that stores additional information #
# such as state normalisation and computes these values                         #
#################################################################################

import gym
import numpy as np
from maths import normalise
from domain_params import get_env_kwargs


class EnvWrapper():

    def __init__(self, env_name, completion_fitness=None, domain_params=None,
                 domain_params_input=False, normalise_state=False,
                 domain_params_low=None, domain_params_high=None):

        self._env_name = env_name
        self._completion_fitness = completion_fitness
        # This is a list of domain parameters to use for each trial
        self._domain_params = domain_params
        self._domain_params_input = domain_params_input
        self._normalise_state = normalise_state
        self._domain_params_low = domain_params_low
        self._domain_params_high = domain_params_high

        self._domain_param = None

    # The trial number determines which of the domain parameters to use
    def make_env(self, trial_num=0, seed=True):

        if self._domain_params is None:
            self._env = gym.make(self._env_name)
        else:
            # Determine correct domain parameter
            self._domain_param = self._domain_params[trial_num]
            env_kwargs = get_env_kwargs(self._env_name, self._domain_param)
            self._env = gym.make(self._env_name, **env_kwargs)

        if seed:
            self._env.seed(108)

    def step(self, actions):
        state, r, done, info = self._env.step(actions)
        state = self._process_state(state)
        return state, r, done, info

    # Applies normalisation and includes domain parameters in state if specified
    def _process_state(self, state):
        if self._normalise_state:
            # Normalise state to [0, 1]
            state = normalise(state, self._env.observation_space.high,
                              self._env.observation_space.low)

        # Add domain parameters to input
        if self._domain_params_input:
            # Normalise domain parameters
            if self._domain_params_low and self._domain_params_high:
                domain_param = normalise([self._domain_param],
                                         self._domain_params_high,
                                         self._domain_params_low)
            # Append domain parameters to state
            state = np.append(state, domain_param)

        return state

    def reset(self):
        return self._process_state(self._env.reset())

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def domain_params(self):
        return self._domain_params

    @property
    def domain_params_input(self):
        return self._domain_params_input

    @property
    def completion_fitness(self):
        return self._completion_fitness

    @property
    def action_space(self):
        return self._env.action_space
