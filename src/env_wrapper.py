#################################################################################
# Provides a wrapper around gym environments that stores additional information #
# such as state normalisation and computes these values                         #
#################################################################################

import gym
import numpy as np
import json
from maths.maths import normalise
from maths.distribution import Distribution
from domain_params import get_env_kwargs


class EnvWrapper():

    def __init__(self, env_name=None, completion_fitness=None,
                 domain_param_distribution=None, domain_params=None,
                 domain_params_input=False, normalise_state=False,
                 domain_params_low=None, domain_params_high=None,
                 env_path=None):

        self._env_name = env_name
        self._completion_fitness = completion_fitness
        # This is distribution from which domain parameters are sampled
        if domain_param_distribution is not None:
            self._domain_param_distribution = Distribution.read(domain_param_distribution)
        # This is a list of domain parameters to use for each trial
        self._domain_params = domain_params
        self._domain_params_input = domain_params_input
        self._normalise_state = normalise_state
        self._domain_params_low = domain_params_low
        self._domain_params_high = domain_params_high

        if env_path:
            self._read(env_path)

        self._domain_param = None

    # The trial number determines which of the domain parameters to use
    def make_env(self, trial_num=0, seed=None):

        if (self._domain_params is None) and (self._domain_param_distribution is None):
            self._env = gym.make(self._env_name)
        else:
            if self._domain_params is not None:
                # Determine correct domain parameter from trial number
                self._domain_param = self._domain_params[trial_num]

            elif self._domain_param_distribution is not None:
                # Sample domain param from distribution
                self._domain_param = self._domain_param_distribution.next()

            env_kwargs = get_env_kwargs(self._env_name, self._domain_param)
            self._env = gym.make(self._env_name, **env_kwargs)

        if seed is not None:
            self._env.seed(seed)

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

    def _read(self, file_path):

        with open(file_path, 'r') as f:
            data = json.load(f)

        env_data = data['env']

        self._env_name = env_data['env_name']
        self._domain_params_input = env_data['domain_params_input']
        self._normalise_state = env_data['normalise_state']
        self._domain_params_low = env_data['domain_params_low']
        self._domain_params_high = env_data['domain_params_high']

    def to_dict(self):

        env_dict = {
            'env_name': self._env_name,
            'completion_fitness': self._completion_fitness,
            'domain_params': self._domain_params,
            'domain_params_input': self._domain_params_input,
            'normalise_state': self._normalise_state,
            'domain_params_low': self._domain_params_low,
            'domain_params_high': self._domain_params_high
        }

        return env_dict

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
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def reward_range(self):
        return self._env.reward_range

    @property
    def metadata(self):
        return self._env.metadata

    @property
    def spec(self):
        return self._env.spec

    def reset(self):
        return self._process_state(self._env.reset())

    def render(self, mode='human'):
        self._env.render(mode=mode)

    def close(self):
        self._env.close()
