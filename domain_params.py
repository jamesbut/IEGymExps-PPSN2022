import numpy as np
import random

def get_kwarg_values(env_name):
    if env_name == 'BipedalWalkerv3':
        return 4.75
    elif env_name == 'MountainCarContinuous-v0':
        return [0.0008]
        #return [0.0012, 0.0012, 0.0012]
        #return [0.0011, 0.0012, 0.0013]
        #return [0.0008, 0.0012, 0.0016]
        #return [0.0014]

#Get env kwargs for particular environment
def get_env_kwargs(randomise, env_name):

    if randomise:

        if env_name == 'BipedalWalker-v3':
            #Range is [2., 6.] - does not include 6.5
            incremented_speeds = np.arange(2., 6.5, 0.5)
            selected_speed = random.choice(incremented_speeds)

            env_kwargs = {
                'speed_knee' : selected_speed
            }

        elif env_name == 'MountainCarContinuous-v0':
            #Range is [0.0020, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025]
            #param_range = np.arange(0.002, 0.0026, 0.0001)

            param_range = [0.0008, 0.0012, 0.0016]
            selected_param = random.choice(param_range)

            env_kwargs = [{
                'power' : selected_param
            }]

        else:
            env_kwargs = None

    else:

        if env_name == 'BipedalWalker-v3':
            env_kwargs = {
                #'speed_knee' : 6.
                'speed_knee' : 4.75
            }

        elif env_name == 'MountainCarContinuous-v0':
            #Create list of env kwargs for different trials for the same organism
            env_kwargs = []
            env_kwarg_vals = get_kwarg_values(env_name)
            for val in env_kwarg_vals:
                env_kwargs.append(
                    {
                        'power' : val
                    }
                )

        else:
            env_kwargs = None

    return env_kwargs

#Get env kwargs as a list
def get_domain_params(env_kwargs, env_name):

    if env_name == 'BipedalWalker-v3':
        return env_kwargs['speed_knee']
    elif env_name == 'MountainCarContinuous-v0':
        kwargs = []
        for val in env_kwargs:
            kwargs.append(val['power'])
        return kwargs
    else:
        return None
