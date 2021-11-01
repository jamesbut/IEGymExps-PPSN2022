from agent import Agent
from deap import creator, base, cma, tools
import evo_utils
import numpy as np
import uuid
import sys
import torch
from data import dump_data, create_exp_dir_name
from domain_params import get_env_kwargs
from model_training import train_ae, train_vae, train_gan
from evo_utils import get_cmaes_centroid
from evaluate import evaluate
from env_wrapper import EnvWrapper
import constants as consts

# Suppress scientific notation
np.set_printoptions(suppress=True)


def evo_run(env_wrapper, dir_path, exp_dir_path):
    '''
    Define neural controller according to environment
    '''
    env_wrapper.make_env()
    state = env_wrapper.reset()

    num_inputs = len(state)
    num_outputs = len(env_wrapper.action_space.high)

    decoder = None
    if consts.USE_DECODER:
        decoder_path = 'generator.pt'
        try:
            decoder = torch.load(decoder_path)
        except IOError:
            print("Could not find requested decoder!!:", decoder_path)

    agent = Agent(num_inputs, num_outputs, consts.NUM_HIDDEN_LAYERS,
                  consts.NEURONS_PER_HIDDEN_LAYER, decoder=decoder,
                  bias=consts.BIAS, w_lb=consts.W_LB, w_ub=consts.W_UB,
                  enforce_wb=consts.ENFORCE_WB)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, agent=agent, env_wrapper=env_wrapper,
                     render=consts.RENDER, avg_fitnesses=True)

    '''
    Define evolutionary algorithm
    '''
    num_genes = agent.genotype_size

    centroid = get_cmaes_centroid(num_genes, sys.argv[:],
                                  dir_path=dir_path, file_name=consts.WINNER_FILE_NAME)

    # Expand gene bounds if gene bound list is only of length 1
    g_lb = consts.G_LB
    g_ub = consts.G_UB
    if len(consts.G_LB) == 1:
        g_lb *= num_genes
    if len(consts.G_UB) == 1:
        g_ub *= num_genes

    strategy = cma.Strategy(centroid=centroid, sigma=consts.INIT_SIGMA,
                            lambda_=consts.LAMBDA, lb_=g_lb, ub_=g_ub)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    '''
    Define execution and logs
    '''
    # np.random.seed(108)

    if consts.PARALLELISE:
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
        toolbox, ngen=consts.NUM_GENS, stats=stats,
        halloffame=hof, completion_fitness=env_wrapper.completion_fitness)

    '''
    Write results to file
    '''
    run_path = exp_dir_path + str(uuid.uuid4()) + '/'

    if ((consts.SAVE_WINNERS_ONLY is False)
         or (consts.SAVE_WINNERS_ONLY is True and complete)):
        agent.genotype = hof[0]
        agent.fitness = hof[0].fitness.values[0]
        g_saved = agent.save(run_path, consts.WINNER_FILE_NAME,
                             env_wrapper, consts.SAVE_IF_WB_EXCEEDED)

        # Save population statistics
        if g_saved:
            dump_data(logbook.select('avg'), run_path, 'mean_fitnesses')
            dump_data(logbook.select('max'), run_path, 'best_fitnesses')

    if consts.PARALLELISE:
        pool.close()


def indv_run(genotype_path, env_name, domain_parameters, render=True):

    # render = False

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

    # Create experiment path
    exp_dir_path = consts.DATA_DIR_PATH + 'python_data'
    exp_dir_name = create_exp_dir_name(exp_dir_path)
    exp_dir_path += '/' + exp_dir_name + '/'

    if (len(sys.argv) == 1) or ('-cmaes_centroid' in sys.argv):

        env_wrapper = EnvWrapper(consts.ENV_NAME, consts.COMPLETION_FITNESS,
                                 consts.DOMAIN_PARAMETERS, consts.DOMAIN_PARAMS_INPUT,
                                 consts.NORMALISE_STATE, consts.DOMAIN_PARAMS_LOW,
                                 consts.DOMAIN_PARAMS_HIGH)

        for i in range(consts.NUM_EVO_RUNS):
            print("Evo run: ", i)
            evo_run(env_wrapper, consts.DATA_DIR_PATH, exp_dir_path)
    else:

        print("Individual run")

        # Genome directory comes from the command line
        indv_dir = sys.argv[1]
        indv_path = consts.DATA_DIR_PATH + indv_dir + '/' + consts.WINNER_FILE_NAME

        indv_run(indv_path, consts.ENV_NAME, consts.DOMAIN_PARAMETERS)


# Some bug in DEAP means that I have to create individual before if __name__ == "__main__"

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

if __name__ == "__main__":
    main()
