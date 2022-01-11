####################################################################
# Code to evaluate network or genome performance on an environment #
####################################################################

import copy


def run(agent, env, render=False):

    reward = 0
    done = False

    if render:
        env.render()
    state = env.reset()

    while not done:

        if render:
            env.render()

        net_out = agent.forward(state)

        # Normalise output between action space bounds
        action_vals = net_out * (env.action_space.high - env.action_space.low) + \
                                 env.action_space.low

        state, r, done, info = env.step(action_vals)

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


# Either pass in a genome and an agent with the required architecture OR
# an agent with the network weights already set
def evaluate(genome=None, agent=None, env_wrapper=None,
             render=False, verbosity=False, avg_fitnesses=False):

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

        if verbosity:
            print("Domain parameters:", env_wrapper.domain_params[trial_num])

        env_wrapper.make_env(trial_num)
        r = run(agent, env_wrapper, render)
        rewards.append(r)

        if verbosity:
            print("Reward: ", r)

    if avg_fitnesses:
        return [sum(rewards) / len(rewards)]
    else:
        return rewards
