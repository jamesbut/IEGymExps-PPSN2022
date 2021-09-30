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

def reset_env(env):

    #Check for pybullet env
    if 'PyBulletEnv' in env_name:
        state = env.reset()
        delete_last_lines(2)
        return state
    else:
        state = env.reset()
        return state

def run(genome=None, num_inputs=None, num_outputs=None,
        num_hidden_layers=None, neurons_per_hidden_layer=None,
        render=False, genotype_dir=None, env_kwargs=None):

    if env_kwargs is not None:
        env = gym.make(env_name, **env_kwargs)
    else:
        env = gym.make(env_name)

    env.seed(108)

    nn = NeuralNetwork(num_inputs, num_outputs,
                       num_hidden_layers, neurons_per_hidden_layer,
                       genotype=genome, genotype_dir=genotype_dir,
                       decoder=use_decoder, bias=bias)

    reward = 0
    done = False

    if render:
        env.render()

    state = env.reset()

    while not done:

        if render:
            env.render()

        net_out = nn.forward(state)

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


def evaluate(genome=None, num_inputs=None, num_outputs=None,
             num_hidden_layers=None, neurons_per_hidden_layer=None,
             render=False, genotype_dir=None, env_kwargs=None,
             verbosity=False):

    reward = 0

    #For a certain number of trials/env arguments
    for kwargs in env_kwargs:
        r = run(genome, num_inputs, num_outputs, num_hidden_layers,
                neurons_per_hidden_layer, render, genotype_dir, kwargs)
        reward += r

        if verbosity:
            print(kwargs)
            print("Reward: ", r)

    #print("Reward before:", reward)
    #Average reward over number of trials
    reward /= len(env_kwargs)
    #print("Reward after:", reward)

    return [reward]


def evo_run(num_inputs, num_outputs, num_hidden_layers, neurons_per_hidden_layer,
            dir_path, file_name, run_num, domain_params):

    print("Domain params:", domain_params)

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

    num_gens = 1
    dump_every = 25
    population, logbook, avg_fitnesses, best_fitnesses, complete = \
        evo_utils.eaGenerateUpdate(toolbox, ngen=num_gens, stats=stats, halloffame=hof,
                                   dump_every=dump_every, dummy_nn=dummy_nn,
                                   completion_fitness=completion_fitness)

    dir_path += str(uuid.uuid4()) + '/'

    #Save best agent
    save_winners_only = False

    if ((save_winners_only is False) or
       (save_winners_only is True and complete)):
        dummy_nn.set_genotype(hof[0])
        dummy_nn.save_genotype(dir_path, file_name, hof[0].fitness.getValues()[0],
                               domain_params)

        #Save population statistics
        dump_data(avg_fitnesses, dir_path, 'mean_fitnesses')
        dump_data(best_fitnesses, dir_path, 'best_fitnesses')

    if parallelise:
        pool.close()

    return dummy_nn

def indv_run(num_inputs, num_outputs,
             num_hidden_layers, neurons_per_hidden_layer,
             genotype=None, genotype_dir=None, env_kwargs=None):

    render = False

    reward = evaluate(genotype, num_inputs, num_outputs,
                      num_hidden_layers, neurons_per_hidden_layer,
                      render, genotype_dir, env_kwargs, verbosity=True)

    print("Reward: ", reward)


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


    randomise_hyperparams = False

    if len(sys.argv)==1:

        num_runs = 1

        #Create experiment path
        exp_dir_name = create_exp_dir_name(dir_path + 'python_data')
        dir_exp_path = dir_path + 'python_data/' + exp_dir_name + '/'

        for i in range(num_runs):
            print("Evo run: ", i)

            env_kwargs = get_env_kwargs(randomise_hyperparams, env_name)

            toolbox.register("evaluate", evaluate,
                             num_inputs=num_inputs, num_outputs=num_outputs,
                             num_hidden_layers=num_hidden_layers,
                             neurons_per_hidden_layer=neurons_per_hidden_layer,
                             render=render, genotype_dir=None, env_kwargs=env_kwargs)

            domain_params = get_domain_params(env_kwargs, env_name)

            winner = evo_run(num_inputs, num_outputs, num_hidden_layers,
                             neurons_per_hidden_layer, dir_exp_path, file_name, i,
                             domain_params)

            #Reset strategy
            strategy = cma.Strategy(centroid=centroid, sigma=init_sigma, lambda_=lambda_,
                                    lb_=lb, ub_=ub)
            toolbox.register("generate", strategy.generate, creator.Individual)
            toolbox.register("update", strategy.update)

        #indv_run(winner.get_weights(), num_inputs, num_outputs, num_hidden_layers,
        #         neurons_per_hidden_layer)

    else:

        print("Individual run")

        env_kwargs = get_env_kwargs(False, env_name)

        #Genome directory comes from the command line
        indv_dir = sys.argv[1]

        indv_full_path = dir_path + 'python_data/' + indv_dir + "/" + file_name

        indv_run(num_inputs, num_outputs, num_hidden_layers,
                 neurons_per_hidden_layer, genotype_dir=indv_full_path,
                 env_kwargs=env_kwargs)


dir_path = "../IndirectEncodingsExperiments/lib/NeuroEvo/data/"
file_name = "best_winner_so_far"

#Some bug in DEAP means that I have to define toolbox before if __name__ == "__main__"
#apparently

#env_name = 'BipedalWalker-v3'
#env_name = 'HalfCheetah-v2'
#env_name = 'LunarLanderContinuous-v2'
#env_name = 'HumanoidPyBulletEnv-v0'
#env_name = 'HalfCheetahPyBulletEnv-v0'
#env_name = 'InvertedDoublePendulum-v2'
env_name = 'MountainCarContinuous-v0'

if 'PyBulletEnv' in env_name:
    import pybulletgym

completion_fitness = None
#completion_fitness = 2.2

dummy_env = gym.make(env_name)
state = dummy_env.reset()

num_inputs = len(state)
num_outputs = len(dummy_env.action_space.high)
num_hidden_layers = 0
neurons_per_hidden_layer = 0
bias=False

render = False
use_decoder = False

dummy_nn = NeuralNetwork(num_inputs, num_outputs, num_hidden_layers,
                         neurons_per_hidden_layer, decoder=use_decoder,
                         bias=bias)
#num_weights = dummy_nn.get_num_weights()
num_genes = dummy_nn.get_genotype_size()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate,
                 num_inputs=num_inputs, num_outputs=num_outputs,
                 num_hidden_layers=num_hidden_layers,
                 neurons_per_hidden_layer=neurons_per_hidden_layer,
                 render=render, genotype_dir=None)

#Initial location of distribution centre
centroid = get_cmaes_centroid(num_genes, sys.argv[:],
                              dir_path=dir_path, file_name=file_name)
#Initial standard deviation of the distribution
init_sigma = 1.0
#Number of children to produce at each generation
#lambda_ = 20 * num_weights
lambda_ = 100

lb = [-10., -10.]
ub = [10., 120.]

strategy = cma.Strategy(centroid=centroid, sigma=init_sigma, lambda_=lambda_,
                        lb_=lb, ub_=ub)

toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

if __name__ == "__main__":
    main()
