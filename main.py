from neural_network import NeuralNetwork
from decoder import Decoder
import gym
from deap import creator, base, cma, tools
import evo_utils
import numpy as np
import uuid
import sys
import random
from data import *
from formatting import *
from domain_params import *
from model_training import *
from evo_utils import get_cmaes_centroid

#Suppress scientific notation
np.set_printoptions(suppress=True)


def run(network, env_name, env_kwargs=None, render=False):

    if env_kwargs is not None:
        env = gym.make(env_name, **env_kwargs)
    else:
        env = gym.make(env_name)

    env.seed(108)

    reward = 0
    done = False

    if render:
        env.render()

    state = env.reset()

    while not done:

        if render:
            env.render()

        net_out = network.forward(state)

        #Normalise output between action space bounds
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


#Either pass in a genome and a network with the required architecture OR
#a network with the weights already set
def evaluate(genome=None, network=None,
             env_name=None, env_kwargs=None, render=False,
             verbosity=False, avg_fitnesses=False):

    if genome is not None:
        network.set_genotype(genome)

    rewards = []

    #For a certain number of trials/env arguments
    for kwargs in env_kwargs:
        r = run(network, env_name, kwargs, render)
        rewards.append(r)

        if verbosity:
            print(kwargs)
            print("Reward: ", r)

    if avg_fitnesses:
        return [sum(rewards) / len(rewards)]
    else:
        return rewards


def evo_run(env_name, completion_fitness, dir_path, file_name):

    '''
    Define neural controller according to environment
    '''
    dummy_env = gym.make(env_name)
    state = dummy_env.reset()

    num_inputs = len(state)
    num_outputs = len(dummy_env.action_space.high)
    num_hidden_layers = 0
    neurons_per_hidden_layer = 0
    bias=False

    #Weight bounds
    w_lb = [-10., -10.]
    w_ub = [10., 120.]
    #Enforce the weight bounds
    #If this is turned off the weight bounds are not applied
    enforce_wb = True
    #If this is turned off the genome is not saved if weight bounds are exceeded
    save_if_wb_exceeded = True

    render = False
    use_decoder = False
    randomise_hyperparams = False

    network = NeuralNetwork(num_inputs, num_outputs, num_hidden_layers,
                            neurons_per_hidden_layer, decoder=use_decoder,
                            bias=bias, w_lb=w_lb, w_ub=w_ub, enforce_wb=enforce_wb)
    env_kwargs = get_env_kwargs(env_name, randomise_hyperparams)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, network=network,
                     env_name=env_name, env_kwargs=env_kwargs, render=render,
                     avg_fitnesses=True)

    domain_params = get_domain_params(env_kwargs, env_name)
    print("Domain params:", domain_params)

    '''
    Define evolutionary algorithm
    '''
    num_genes = network.get_genotype_size()

    centroid = get_cmaes_centroid(num_genes, sys.argv[:],
                                  dir_path=dir_path, file_name=file_name)
    #Initial standard deviation of the distribution
    init_sigma = 1.0
    #Number of children to produce at each generation
    #lambda_ = 20 * num_weights
    lambda_ = 100
    num_gens = 100

    #Gene bounds
    g_lb = [-10., -10.]
    g_ub = [10., 120.]

    strategy = cma.Strategy(centroid=centroid, sigma=init_sigma, lambda_=lambda_,
                            lb_=g_lb, ub_=g_ub)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    '''
    Define execution and logs
    '''

    #np.random.seed(108)

    parallelise = True
    if parallelise:
        import multiprocessing

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    num_genomes_in_hof = 1
    hof = evo_utils.HallOfFamePriorityYoungest(num_genomes_in_hof)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    '''
    Run evolutionary algorithm
    '''
    population, logbook, avg_fitnesses, best_fitnesses, complete = \
        evo_utils.eaGenerateUpdate(toolbox, ngen=num_gens, stats=stats, halloffame=hof,
                                   completion_fitness=completion_fitness)

    dir_path += str(uuid.uuid4()) + '/'

    '''
    Write results to file
    '''
    save_winners_only = False

    if ((save_winners_only is False) or
       (save_winners_only is True and complete)):
        network.set_genotype(hof[0])
        g_saved = network.save_genotype(dir_path, file_name,
                                        hof[0].fitness.getValues()[0],
                                        domain_params, save_if_wb_exceeded)

        #Save population statistics
        if g_saved:
            dump_data(avg_fitnesses, dir_path, 'mean_fitnesses')
            dump_data(best_fitnesses, dir_path, 'best_fitnesses')

    if parallelise:
        pool.close()


def indv_run(genotype_dir, env_name, render=True):

    #render = False

    env_kwargs = get_env_kwargs(env_name, randomise=False)

    network = NeuralNetwork(genotype_dir=genotype_dir)
    rewards = evaluate(network=network,
                       env_name=env_name, env_kwargs=env_kwargs, render=render,
                       verbosity=True)

    print("Rewards: ", rewards)
    print("Mean reward:", sum(rewards) / len(rewards))

    return rewards


def main():

    ae_train = False
    if ae_train:
        train_ae(sys.argv[1])
        return

    vae_train = False
    if vae_train:
        train_vae(sys.argv[1])
        return

    gan_train = False
    if gan_train:
        train_gan(sys.argv[1])
        return

    '''
    Define environment
    '''

    #env_name = 'BipedalWalker-v3'
    #env_name = 'HalfCheetah-v2'
    #env_name = 'LunarLanderContinuous-v2'
    #env_name = 'HumanoidPyBulletEnv-v0'
    #env_name = 'HalfCheetahPyBulletEnv-v0'
    #env_name = 'InvertedDoublePendulum-v2'
    env_name = 'MountainCarContinuous-v0'

    if 'PyBulletEnv' in env_name:
        import pybulletgym

    #completion_fitness = None
    completion_fitness = 2.15

    dir_path = "../IndirectEncodingsExperiments/lib/NeuroEvo/data/"
    file_name = "best_winner_so_far"

    if (len(sys.argv)==1) or ('-cmaes_centroid' in sys.argv):

        num_runs = 2

        #Create experiment path
        exp_dir_name = create_exp_dir_name(dir_path + 'python_data')
        dir_exp_path = dir_path + 'python_data/' + exp_dir_name + '/'

        for i in range(num_runs):
            print("Evo run: ", i)
            evo_run(env_name, completion_fitness, dir_exp_path, file_name)

    else:

        print("Individual run")

        #Genome directory comes from the command line
        indv_dir = sys.argv[1]
        indv_full_path = dir_path + '/' + indv_dir + "/" + file_name

        indv_run(indv_full_path, env_name)


#Some bug in DEAP means that I have to create individual before if __name__ == "__main__"

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

if __name__ == "__main__":
    main()
