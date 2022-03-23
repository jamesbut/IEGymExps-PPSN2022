####################################################################
# Code to evaluate network or genome performance on an environment #
####################################################################

import copy
import numpy as np
import time


def render_fn(env, render: bool, fps: float):
    if render:
        if fps is not None:
            time.sleep(1. / float(fps))
        env.render()


def run(agent, env, render=False, fps: float = None, verbosity=0):

    reward = 0
    done = False

    render_fn(env, render, fps)
    state = env.reset()

    while not done:

        render_fn(env, render, fps)

        net_out = agent.forward(state)

        # If action space is discrete choose arg max of network output
        if env.discrete_action_space:
            action_vals = np.argmax(net_out)
        # If action space is an array of floats
        else:
            # Normalise output between action space bounds
            action_vals = net_out * (env.action_space.high - env.action_space.low) + \
                                     env.action_space.low

        state, r, done, info = env.step(action_vals)

        reward += r

        if verbosity > 1:
            print("Net out: ", net_out)
            print("Action vals: ", action_vals)
            print("State: ", state)
            print("Reward: ", r)
            print("Total reward: ", reward)

    # Render final frame
    render_fn(env, render, fps)

    env.close()

    return reward


# Either pass in a genome and an agent with the required architecture OR
# an agent with the network weights already set
def evaluate(genome=None, agent=None, env_wrapper=None, render: bool = False,
             fps: float = None, verbosity: int = 0, avg_fitnesses=False, env_seed=None):

    env_wrapper = copy.deepcopy(env_wrapper)

    if genome is not None:
        agent.genotype = genome

    rewards = []

    # Only 1 trial if a domain param distrbution is given
    num_trials = 1
    # The number of trials is taken as the number of domain paramaters in EnvWrapper if
    # given
    if env_wrapper.domain_params:
        num_trials = len(env_wrapper.domain_params)

    # Run trials
    for trial_num in range(num_trials):

        if verbosity > 0 and env_wrapper.domain_params:
            print("Domain parameters:", env_wrapper.domain_params[trial_num])

        env_wrapper.make_env(trial_num, env_seed)
        r = run(agent, env_wrapper, render, fps, verbosity)
        rewards.append(r)

        if verbosity > 0:
            print("Reward: ", r)

    if avg_fitnesses:
        return [sum(rewards) / len(rewards)]
    else:
        return rewards
