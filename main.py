from deap import creator, base, cma, tools
import evo_utils
import numpy as np
import uuid
import sys
from agent import Agent
from data import dump_list, dump_json, read_configs
from model_training import train_generative_model
from evo_utils import get_cmaes_centroid
from evaluate import evaluate
from env_wrapper import EnvWrapper
from neural_network import NeuralNetwork

# Suppress scientific notation
np.set_printoptions(suppress=True)


def evo_run(config, exp_dir_path):

    # Environment
    env_wrapper = EnvWrapper(config['env']['name'],
                             config['env']['completion_fitness'],
                             config['env'].get('domain_param_distribution', None),
                             config['env'].get('domain_params', None),
                             config['env']['domain_params_input'],
                             config['env']['normalise_state'],
                             config['env']['domain_params_low'],
                             config['env']['domain_params_high'])

    # Read decoder for evolution if specified
    decoder = None
    if config['ie']['use_decoder']:
        try:
            decoder = NeuralNetwork(file_path=config['ie']['decoder_path'])
        except IOError:
            print("Could not find requested decoder for evolution:",
                  config['ie']['decoder_path'])

    # Create agent
    env_wrapper.make_env()
    state = env_wrapper.reset()

    num_inputs = len(state)
    num_outputs = len(env_wrapper.action_space.high)

    agent = Agent(num_inputs, num_outputs,
                  config['controller']['num_hidden_layers'],
                  config['controller']['neurons_per_hidden_layer'],
                  config['controller']['hidden_layer_activ_func'],
                  config['controller']['final_layer_activ_func'],
                  config['controller']['bias'],
                  config['controller']['w_lb'], config['controller']['w_ub'],
                  config['controller']['enforce_wb'], decoder=decoder)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, agent=agent, env_wrapper=env_wrapper,
                     render=config['optimiser']['render'], avg_fitnesses=True)

    # Define evolutionary algorithm
    num_genes = agent.genotype_size

    centroid = get_cmaes_centroid(num_genes, sys.argv[:],
                                  dir_path=config['logging']['data_dir_path'],
                                  file_name=config['logging']['winner_file_name'])

    # Expand gene bounds if gene bound list is only of length 1
    g_lb = config['optimiser']['g_lb']
    if len(config['optimiser']['g_lb']) == 1:
        g_lb = config['optimiser']['g_lb'] * num_genes
    g_ub = config['optimiser']['g_ub']
    if len(config['optimiser']['g_ub']) == 1:
        g_ub = config['optimiser']['g_ub'] * num_genes

    strategy = cma.Strategy(centroid=centroid,
                            sigma=config['optimiser']['cmaes']['init_sigma'],
                            lambda_=config['optimiser']['cmaes']['lambda'],
                            lb_=g_lb, ub_=g_ub)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # Define execution and logs

    # np.random.seed(108)

    if config['optimiser']['parallelise']:
        import multiprocessing

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    hof = tools.HallOfFame(maxsize=1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run evolutionary algorithm
    population, logbook, complete = evo_utils.eaGenerateUpdate(
        toolbox, ngen=config['optimiser']['num_gens'], stats=stats,
        halloffame=hof, completion_fitness=env_wrapper.completion_fitness)

    # Write results to file
    run_path = exp_dir_path + str(uuid.uuid4()) + '/'

    if ((config['logging']['save_winners_only'] is False)
         or (config['logging']['save_winners_only'] is True and complete)):
        agent.genotype = hof[0]
        agent.fitness = hof[0].fitness.values[0]
        g_saved = agent.save(run_path, config['logging']['winner_file_name'],
                             env_wrapper, config['logging']['save_if_wb_exceeded'])

        # Save population statistics
        if g_saved:
            dump_list(logbook.select('avg'), run_path, 'mean_fitnesses')
            dump_list(logbook.select('max'), run_path, 'best_fitnesses')

    if config['optimiser']['parallelise']:
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


def main(argv, config):

    # Train decoder
    if '-train_decoder' in argv:

        # Parse command line for information on how to train the decoder
        td_index = argv.index('-train_decoder')
        # Read in the type of model: 'gan', 'ae' or 'vae'
        gen_model_type = argv[td_index + 1]
        # Give training data directory to train model with
        gen_model_train_data_exp_dir = argv[td_index + 2]

        # Train generative model
        train_generative_model(gen_model_type, configs['ie']['code_size'],
                               config['ie']['num_hidden_layers'],
                               config['ie']['neurons_per_hidden_layer'],
                               config['ie']['num_epochs'], config['ie']['batch_size'],
                               config['logging']['data_dir_path']
                               + gen_model_train_data_exp_dir,
                               config['logging']['decoder_path'],
                               config['logging']['winner_file_name'])

    # Evolutionary run
    elif '-evo_run' in argv:

        # Create experiment path
        exp_dir_path = config['logging']['data_dir_path'] \
            + config['logging']['exp_dir_name'] + '/'

        # Run experiment
        for i in range(config['optimiser']['num_runs']):
            print("Evo run: ", i)
            evo_run(config, exp_dir_path)

        # Dump configs
        dump_json(exp_dir_path + 'experiment.json', config)

    # Individual run
    else:

        print("Individual run")

        # Agent directory comes from the command line
        indv_dir = argv[1]
        indv_path = config['logging']['data_dir_path'] + indv_dir + '/' \
                    + config['logging']['winner_file_name']

        indv_run(indv_path, config['env']['domain_params'])


# Some bug in DEAP means that I have to create individual before
# if __name__ == "__main__"

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

if __name__ == "__main__":

    # Read config or config group from command line
    configs = read_configs(sys.argv)

    # Run for different configuration settings
    for config in configs:
        print('Config:', config['logging']['config_file_path'])
        main(sys.argv, config)
