import numpy as np
import random


# Get env kwargs from list of domain parameters
def get_env_kwargs(env_name, domain_param, randomise=False):

    if randomise:

        if env_name == 'BipedalWalker-v3':
            # Range is [2., 6.] - does not include 6.5
            incremented_speeds = np.arange(2., 6.5, 0.5)
            selected_speed = random.choice(incremented_speeds)

            env_kwargs = {
                'speed_knee': selected_speed
            }

        elif env_name == 'MountainCarContinuous-v0':
            # Range is [0.0020, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025]
            # param_range = np.arange(0.002, 0.0026, 0.0001)

            param_range = [0.0008, 0.0012, 0.0016]
            selected_param = random.choice(param_range)

            env_kwargs = [{
                'power': selected_param
            }]

        else:
            env_kwargs = None

    else:

        env_kwargs = {
            _get_param_string(env_name): domain_param
        }

    return env_kwargs


# Get domain parameter value from env_kwarg dictionary
def get_domain_param_from_env_kwarg(env_kwarg, env_name):
    return env_kwarg[_get_param_string(env_name)]


# Get list of domain parameter values from env_kwarg dictionaries
def get_domain_params_from_env_kwargs(env_kwargs, env_name):
    params = []
    for env_kwarg in env_kwargs:
        params.append(get_domain_param_from_env_kwarg(env_kwarg))
    return params


# Get string for the domain parameter of interest
def _get_param_string(env_name):

    if env_name == 'BipedalWalker-v3':
        return 'speed_knee'
    elif env_name == 'MountainCarContinuous-v0':
        return 'power'
    elif env_name == 'LunarLanderContinuous-v2':
        return 'scale'
    elif env_name == 'FrozenLake-v0':
        return 'goal_pos'
