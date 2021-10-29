####################################################################
# Code to evaluate network or genome performance on an environment #
####################################################################

import gym
import numpy as np
from domain_params import get_domain_param_from_env_kwarg
from maths import normalise


# Apply operations to state before passing through network
def build_state(state, network, env, env_name, env_kwargs):

    if network.normalise_state:
        # Normalise state to [0, 1]
        state = normalise(state, env.observation_space.high, env.observation_space.low)

    # Add domain parameters to input
    if network.domain_params_input:
        domain_param = get_domain_param_from_env_kwarg(env_kwargs, env_name)
        # Normalise domain parameters
        if network.norm_domain_params_low and network.norm_domain_params_high:
            domain_param = normalise([domain_param],
                                     network.norm_domain_params_high,
                                     network.norm_domain_params_low)
        # Append domain parameters to state
        state = np.append(state, domain_param)

    return state


def run(network, env_name, run_num, env_kwargs=None, render=False):

    if env_kwargs is not None:
        env = gym.make(env_name, **env_kwargs)
    else:
        env = gym.make(env_name)

    env.seed(108)

    reward = 0
    done = False

    if render:
        env.render()

    state = build_state(env.reset(), network, env, env_name, env_kwargs)

    while not done:

        if render:
            env.render()

        net_out = network.forward(state)

        # Normalise output between action space bounds
        action_vals = net_out * (env.action_space.high - env.action_space.low) + \
                      env.action_space.low

        state, r, done, info = env.step(action_vals)
        state = build_state(state, network, env, env_name, env_kwargs)

        quit()

        reward += r

        '''
        print("Net out: ", net_out)
        print("Action vals: ", action_vals)
        print("State: ", state)
        print("Reward: ", r)
        print("Total reward: ", reward)
        '''

    env.close()

    return reward


# Either pass in a genome and a network with the required architecture OR
# a network with the weights already set
def evaluate(genome=None, network=None,
             env_name=None, env_kwargs=None, render=False,
             verbosity=False, avg_fitnesses=False):

    if genome is not None:
        network.genotype = genome

    rewards = []

    # For a certain number of trials/env arguments
    for run_num, kwargs in enumerate(env_kwargs):
        r = run(network, env_name, run_num, kwargs, render)
        rewards.append(r)

        if verbosity:
            print(kwargs)
            print("Reward: ", r)

    if avg_fitnesses:
        return [sum(rewards) / len(rewards)]
    else:
        return rewards
