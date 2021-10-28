from neural_network import NeuralNetwork
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
from constants import *
from evaluate import *

#Suppress scientific notation
np.set_printoptions(suppress=True)


def evo_run(env_name, completion_fitness, dir_path, exp_dir_path):

    '''
    Define neural controller according to environment
    '''
    dummy_env = gym.make(env_name)
    state = dummy_env.reset()

    num_inputs = len(state)
    num_outputs = len(dummy_env.action_space.high)

    decoder = None
    if USE_DECODER:
        decoder_path = 'generator.pt'
        try:
            decoder = torch.load(decoder_path)
        except IOError:
            print("Could not find requested decoder!!:", decoder_path)

    network = NeuralNetwork(num_inputs, num_outputs, NUM_HIDDEN_LAYERS,
                            NEURONS_PER_HIDDEN_LAYER, decoder=decoder,
                            bias=BIAS, w_lb=W_LB, w_ub=W_UB, enforce_wb=ENFORCE_WB,
                            domain_params_input=DOMAIN_PARAMS_INPUT)

    env_kwargs = get_env_kwargs(env_name, DOMAIN_PARAMETERS, RANDOMISE_HYPERPARAMETERS)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, network=network,
                     env_name=env_name, env_kwargs=env_kwargs, render=RENDER,
                     avg_fitnesses=True)

    '''
    Define evolutionary algorithm
    '''
    num_genes = network.genotype_size

    centroid = get_cmaes_centroid(num_genes, sys.argv[:],
                                  dir_path=dir_path, file_name=WINNER_FILE_NAME)

    #Expand gene bounds if gene bound list is only of length 1
    g_lb = G_LB
    g_ub = G_UB
    if len(G_LB) == 1:
        g_lb *= num_genes
    if len(G_UB) == 1:
        g_ub *= num_genes

    strategy = cma.Strategy(centroid=centroid, sigma=INIT_SIGMA, lambda_=LAMBDA,
                            lb_=g_lb, ub_=g_ub)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    '''
    Define execution and logs
    '''
    #np.random.seed(108)

    if PARALLELISE:
        import multiprocessing

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    hof = tools.HallOfFame(maxsize=1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    '''
    Run evolutionary algorithm
    '''
    population, logbook, complete = evo_utils.eaGenerateUpdate(
            toolbox, ngen=NUM_GENS, stats=stats,
            halloffame=hof, completion_fitness=completion_fitness)

    '''
    Write results to file
    '''
    run_path = exp_dir_path + str(uuid.uuid4()) + '/'

    if ((SAVE_WINNERS_ONLY is False) or
       (SAVE_WINNERS_ONLY is True and complete)):
        network.genotype = hof[0]
        g_saved = network.save(run_path, WINNER_FILE_NAME,
                               hof[0].fitness.values[0],
                               DOMAIN_PARAMETERS, SAVE_IF_WB_EXCEEDED)

        #Save population statistics
        if g_saved:
            dump_data(logbook.select('avg'), run_path, 'mean_fitnesses')
            dump_data(logbook.select('max'), run_path, 'best_fitnesses')

    if PARALLELISE:
        pool.close()


def indv_run(genotype_path, env_name, domain_parameters, render=True):

    #render = False

    env_kwargs = get_env_kwargs(env_name, domain_parameters)

    network = NeuralNetwork(genotype_path=genotype_path)
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

    #Create experiment path
    exp_dir_path = DATA_DIR_PATH + 'python_data'
    exp_dir_name = create_exp_dir_name(exp_dir_path)
    exp_dir_path += '/' + exp_dir_name + '/'

    if (len(sys.argv)==1) or ('-cmaes_centroid' in sys.argv):

        for i in range(NUM_EVO_RUNS):
            print("Evo run: ", i)
            evo_run(ENV_NAME, COMPLETION_FITNESS, DATA_DIR_PATH, exp_dir_path)

    else:

        print("Individual run")

        #Genome directory comes from the command line
        indv_dir = sys.argv[1]
        indv_path = DATA_DIR_PATH + indv_dir + '/' + WINNER_FILE_NAME

        indv_run(indv_path, ENV_NAME, DOMAIN_PARAMETERS)


#Some bug in DEAP means that I have to create individual before if __name__ == "__main__"

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

if __name__ == "__main__":
    main()
