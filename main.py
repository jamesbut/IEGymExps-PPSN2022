from deap import creator, base, cma, tools
import evo_utils
import numpy as np
import uuid
import sys
import copy
from agent import Agent
from data import dump_list, dump_json, read_configs
from evo_utils import get_cmaes_centroid, expand_bound
from evaluate import evaluate
from env_wrapper import EnvWrapper
from neural_network import NeuralNetwork
from command_line import parse_axis_limits, parse_test_decoder

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
            decoder_file_path = config['ie']['dump_model_dir'] + '/' + \
                                config['ie']['name'] + '_' + \
                                str(config['ie']['decoder_file_num']) + '.pt'
            decoder = NeuralNetwork(file_path=decoder_file_path)
        except IOError:
            print("Could not find requested decoder for evolution:", decoder_file_path)

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
                  config['controller'].get('w_lb', None),
                  config['controller'].get('w_ub', None),
                  config['controller'].get('enforce_wb', False),
                  decoder=decoder)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, agent=agent, env_wrapper=env_wrapper,
                     render=config['optimiser']['render'], avg_fitnesses=True,
                     env_seed=config['env'].get('seed', None))

    # Define evolutionary algorithm
    num_genes = agent.genotype_size

    # Determine initial centroid of CMAES
    config['optimiser']['cmaes']['centroid'] = \
        get_cmaes_centroid(num_genes, config['optimiser']['cmaes'],
                           dir_path=config['logging']['data_dir_path'],
                           file_name=config['logging']['winner_file_name'])

    # Expand gene bounds
    config['optimiser']['cmaes']['lb'] = \
        expand_bound(config['optimiser'].get('g_lb', None), num_genes)
    config['optimiser']['cmaes']['ub'] = \
        expand_bound(config['optimiser'].get('g_ub', None), num_genes)

    strategy = cma.Strategy(**config['optimiser']['cmaes'])

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # Define execution and logs
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

    # Expand pheno bounds
    num_weights = len(agent.weights)
    p_lb = expand_bound(config['optimiser'].get('p_lb', None), num_weights)
    p_ub = expand_bound(config['optimiser'].get('p_ub', None), num_weights)

    # Run evolutionary algorithm
    population, logbook, complete = evo_utils.eaGenerateUpdate(
        toolbox, ngen=config['optimiser']['num_gens'], stats=stats,
        halloffame=hof, completion_fitness=env_wrapper.completion_fitness,
        quit_domain_when_complete=config['optimiser']['quit_domain_when_complete'],
        decoder=decoder, pop_size=config['optimiser']['cmaes']['lambda'],
        p_lb=p_lb, p_ub=p_ub)

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

    agent = Agent(agent_path=agent_path)
    env_wrapper = EnvWrapper(env_path=agent_path, domain_params=domain_params)

    rewards = evaluate(agent=agent, env_wrapper=env_wrapper, render=render,
                       verbosity=True)

    print("Rewards: ", rewards)
    print("Mean reward:", sum(rewards) / len(rewards))

    return rewards


def main(argv, config):

    if '-train_decoder' in argv:

        from generative_models.model_training import train_generative_model

        # Train generative models
        for _ in range(config['ie']['num_trains']):
            train_generative_model(
                config['ie']['name'], config['ie']['code_size'],
                config['ie']['num_hidden_layers'],
                config['ie']['neurons_per_hidden_layer'],
                config['ie']['num_epochs'], config['ie']['batch_size'],
                config['logging']['data_dir_path']
                + config['ie']['training_data_dir'],
                config['ie']['dump_model_dir'],
                config['logging']['winner_file_name'],
                config['ie']['optimiser'])

    elif '-test_decoder' in argv:

        from generative_models.model_testing import test_decoder

        train_data_path = config['logging']['data_dir_path'] \
                          + config['ie']['training_data_dir']

        # Get decoder from command line
        try:
            decoder_type, decoder_num = parse_test_decoder(argv)
        except IndexError:
            return

        # Get axis limits from command line
        plot_axis_lb, plot_axis_ub = parse_axis_limits(argv)

        # Test decoder
        test_decoder(config['ie']['dump_model_dir'],
                     decoder_type,
                     decoder_num,
                     train_data_path,
                     config['logging']['winner_file_name'],
                     plot_axis_lb=plot_axis_lb,
                     plot_axis_ub=plot_axis_ub,
                     colour_params=True if '-colour_params' in argv else False)

    # Evolutionary run
    elif '-evo_run' in argv:

        import time

        # Create experiment path
        exp_dir_path = config['logging']['data_dir_path'] \
            + config['logging']['exp_dir_name'] + '/'

        # Run experiment
        for i in range(config['execution']['num_runs']):

            print("Evo run: ", i)

            start = time.time()
            evo_run(copy.deepcopy(config), exp_dir_path)
            end = time.time()

            print('Time taken for evolution: {} seconds\n'.format(end - start))

        # Dump configs
        dump_json(exp_dir_path + 'experiment.json', config)

    # Individual run
    else:

        print("Individual run")

        # Agent directory comes from the command line
        indv_dir = argv[1]
        indv_path = config['logging']['data_dir_path'] + indv_dir + '/' \
                    + config['logging']['winner_file_name'] + '.json'

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
