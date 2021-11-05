from deap import creator, base, cma, tools
import evo_utils
import numpy as np
import uuid
import sys
from agent import Agent
from data import dump_list, create_exp_dir_name
from model_training import train_generative_model
from evo_utils import get_cmaes_centroid
from evaluate import evaluate
from env_wrapper import EnvWrapper
from neural_network import NeuralNetwork
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

    # Read decoder for evolution if specified
    decoder = None
    if consts.USE_DECODER:
        try:
            decoder = NeuralNetwork(file_path=consts.DECODER_PATH)
        except IOError:
            print("Could not find requested decoder for evolution:",
                  consts.DECODER_PATH)

    agent = Agent(num_inputs, num_outputs,
                  consts.NUM_HIDDEN_LAYERS, consts.NEURONS_PER_HIDDEN_LAYER,
                  consts.HIDDEN_LAYER_ACTIV_FUNC, consts.FINAL_LAYER_ACTIV_FUNC,
                  bias=consts.BIAS, w_lb=consts.W_LB, w_ub=consts.W_UB,
                  enforce_wb=consts.ENFORCE_WB, decoder=decoder)

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
            dump_list(logbook.select('avg'), run_path, 'mean_fitnesses')
            dump_list(logbook.select('max'), run_path, 'best_fitnesses')

    if consts.PARALLELISE:
        pool.close()


def indv_run(agent_path, domain_params, render=True):

    # render = False

    agent = Agent(agent_path=agent_path)
    env_wrapper = EnvWrapper(env_path=agent_path, domain_params=domain_params)

    rewards = evaluate(agent=agent, env_wrapper=env_wrapper, render=render,
                       verbosity=True)

    print("Rewards: ", rewards)
    print("Mean reward:", sum(rewards) / len(rewards))

    return rewards


def main(argv):

    # Train decoder
    if '-train_decoder' in argv:

        # Parse command line for information on how to train the decoder
        td_index = argv.index('-train_decoder')
        # Read in the type of model: 'gan', 'ae' or 'vae'
        gen_model_type = argv[td_index + 1]
        # Give training data directory to train model with
        gen_model_train_data_exp_dir = argv[td_index + 2]

        # Train generative model
        train_generative_model(gen_model_type, consts.CODE_SIZE,
                               consts.D_NUM_HIDDEN_LAYERS,
                               consts.D_NEURONS_PER_HIDDEN_LAYER,
                               consts.NUM_EPOCHS, consts.BATCH_SIZE,
                               consts.DATA_DIR_PATH + gen_model_train_data_exp_dir,
                               consts.DECODER_PATH, consts.WINNER_FILE_NAME)

        return

    # Create experiment path
    exp_dir_name = create_exp_dir_name(consts.DATA_DIR_PATH)
    exp_dir_path = consts.DATA_DIR_PATH + exp_dir_name + '/'

    # Evolutionary run
    if (len(argv) == 1) or ('-cmaes_centroid' in argv):

        env_wrapper = EnvWrapper(consts.ENV_NAME, consts.COMPLETION_FITNESS,
                                 consts.DOMAIN_PARAMETERS, consts.DOMAIN_PARAMS_INPUT,
                                 consts.NORMALISE_STATE, consts.DOMAIN_PARAMS_LOW,
                                 consts.DOMAIN_PARAMS_HIGH)

        for i in range(consts.NUM_EVO_RUNS):
            print("Evo run: ", i)
            evo_run(env_wrapper, consts.DATA_DIR_PATH, exp_dir_path)

    # Individual run
    else:

        print("Individual run")

        # Genome directory comes from the command line
        indv_dir = argv[1]
        indv_path = consts.DATA_DIR_PATH + indv_dir + '/' + consts.WINNER_FILE_NAME

        indv_run(indv_path, consts.DOMAIN_PARAMETERS)


# Some bug in DEAP means that I have to create individual before
# if __name__ == "__main__"

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

if __name__ == "__main__":
    main(sys.argv)
