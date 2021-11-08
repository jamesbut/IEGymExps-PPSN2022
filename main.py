from deap import creator, base, cma, tools
import evo_utils
import numpy as np
import uuid
import sys
from agent import Agent
from data import dump_list, create_exp_dir_name, dump_json, read_json
from model_training import train_generative_model
from evo_utils import get_cmaes_centroid
from evaluate import evaluate
from env_wrapper import EnvWrapper
from neural_network import NeuralNetwork

# Suppress scientific notation
np.set_printoptions(suppress=True)


def evo_run(configs, exp_dir_path):

    # Environment
    env_wrapper = EnvWrapper(configs['env']['name'],
                             configs['env']['completion_fitness'],
                             configs['env']['domain_params'],
                             configs['env']['domain_params_input'],
                             configs['env']['normalise_state'],
                             configs['env']['domain_params_low'],
                             configs['env']['domain_params_high'])

    # Read decoder for evolution if specified
    decoder = None
    if configs['ie']['use_decoder']:
        try:
            decoder = NeuralNetwork(file_path=configs['ie']['decoder_path'])
        except IOError:
            print("Could not find requested decoder for evolution:",
                  configs['ie']['decoder_path'])

    # Create agent
    env_wrapper.make_env()
    state = env_wrapper.reset()

    num_inputs = len(state)
    num_outputs = len(env_wrapper.action_space.high)

    agent = Agent(num_inputs, num_outputs,
                  configs['controller']['num_hidden_layers'],
                  configs['controller']['neurons_per_hidden_layer'],
                  configs['controller']['hidden_layer_activ_func'],
                  configs['controller']['final_layer_activ_func'],
                  configs['controller']['bias'],
                  configs['controller']['w_lb'], configs['controller']['w_ub'],
                  configs['controller']['enforce_wb'], decoder=decoder)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, agent=agent, env_wrapper=env_wrapper,
                     render=configs['optimiser']['render'], avg_fitnesses=True)

    # Define evolutionary algorithm
    num_genes = agent.genotype_size

    centroid = get_cmaes_centroid(num_genes, sys.argv[:],
                                  dir_path=configs['logging']['data_dir_path'],
                                  file_name=configs['logging']['winner_file_name'])

    # Expand gene bounds if gene bound list is only of length 1
    if len(configs['optimiser']['g_lb']) == 1:
        configs['optimiser']['g_lb'] *= num_genes
    if len(configs['optimiser']['g_ub']) == 1:
        configs['optimiser']['g_ub'] *= num_genes

    strategy = cma.Strategy(centroid=centroid,
                            sigma=configs['optimiser']['cmaes']['init_sigma'],
                            lambda_=configs['optimiser']['cmaes']['lambda'],
                            lb_=configs['optimiser']['g_lb'],
                            ub_=configs['optimiser']['g_ub'])

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # Define execution and logs

    # np.random.seed(108)

    if configs['optimiser']['parallelise']:
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
        toolbox, ngen=configs['optimiser']['num_gens'], stats=stats,
        halloffame=hof, completion_fitness=env_wrapper.completion_fitness)

    # Write results to file
    run_path = exp_dir_path + str(uuid.uuid4()) + '/'

    if ((configs['logging']['save_winners_only'] is False)
         or (configs['logging']['save_winners_only'] is True and complete)):
        agent.genotype = hof[0]
        agent.fitness = hof[0].fitness.values[0]
        g_saved = agent.save(run_path, configs['logging']['winner_file_name'],
                             env_wrapper, configs['logging']['save_if_wb_exceeded'])

        # Save population statistics
        if g_saved:
            dump_list(logbook.select('avg'), run_path, 'mean_fitnesses')
            dump_list(logbook.select('max'), run_path, 'best_fitnesses')

    if configs['optimiser']['parallelise']:
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


def main(argv, configs):

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
                               configs['ie']['num_hidden_layers'],
                               configs['ie']['neurons_per_hidden_layer'],
                               configs['ie']['num_epochs'], configs['ie']['batch_size'],
                               configs['logging']['data_dir_path']
                               + gen_model_train_data_exp_dir,
                               configs['logging']['decoder_path'],
                               configs['logging']['winner_file_name'])

        return

    # Evolutionary run
    if (len(argv) == 1) or ('-cmaes_centroid' in argv):

        # Create experiment path
        exp_dir_name = create_exp_dir_name(configs['logging']['data_dir_path'])
        exp_dir_path = configs['logging']['data_dir_path'] + exp_dir_name + '/'

        # Run experiment
        for i in range(configs['optimiser']['num_runs']):
            print("Evo run: ", i)
            evo_run(configs, exp_dir_path)

        # Dump configs
        dump_json(exp_dir_path + 'experiment.json', configs)

    # Individual run
    else:

        print("Individual run")

        # Agent directory comes from the command line
        indv_dir = argv[1]
        indv_path = configs['data_dir_path'] + indv_dir + '/' \
                    + configs['logging']['winner_file_name']

        indv_run(indv_path, configs['env']['domain_params'])


# Some bug in DEAP means that I have to create individual before
# if __name__ == "__main__"

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

if __name__ == "__main__":
    default_configs = read_json('configs/default.json')
    main(sys.argv, default_configs)
