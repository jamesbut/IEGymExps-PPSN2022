from neural_network import NeuralNetwork
import gym
from deap import creator, base, cma, tools
import evo_utils
import numpy as np
import uuid
import sys
import random

#Suppress scientific notation
np.set_printoptions(suppress=True)

def evaluate(genome, num_inputs, num_outputs,
             num_hidden_layers, neurons_per_hidden_layer,
             render=False, genotype_dir=None, env_kwargs=None):

    env = gym.make("BipedalWalker-v3", **env_kwargs)

    env.seed(108)

    nn = NeuralNetwork(num_inputs, num_outputs,
                       num_hidden_layers, neurons_per_hidden_layer,
                       genotype_dir=genotype_dir)

    if genotype_dir is None:
        nn.set_weights(genome)

    reward = 0
    done = False

    state = env.reset()
    #print("State: ", state)
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

    return [reward]


def evo_run(num_inputs, num_outputs, num_hidden_layers, neurons_per_hidden_layer,
            dir_path, file_name, run_num, domain_params):

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

    num_gens = 100
    dump_every = 25
    population, logbook = evo_utils.eaGenerateUpdate(
        toolbox, ngen=num_gens, stats=stats, halloffame=hof,
        dump_every=dump_every, dummy_nn=dummy_nn)

    #Save best agent
    dir_path += str(uuid.uuid4()) + '/'
    dummy_nn.set_weights(hof[0])
    dummy_nn.save_genotype(dir_path, file_name, hof[0].fitness.getValues()[0],
                           [domain_params])

    if parallelise:
        pool.close()

    return dummy_nn

def indv_run(num_inputs, num_outputs,
             num_hidden_layers, neurons_per_hidden_layer,
             genotype=None, genotype_dir=None):

    render = False

    reward = evaluate(genotype, num_inputs, num_outputs,
                      num_hidden_layers, neurons_per_hidden_layer,
                      render, genotype_dir)

    print("Reward: ", reward)

def main():

    dir_path = "../IndirectEncodingsExperiments/lib/NeuroEvo/data/python_training/"
    file_name = "best_winner_so_far"

    dummy_env = gym.make("BipedalWalker-v3")
    state = dummy_env.reset()

    num_inputs = len(state)
    num_outputs = len(dummy_env.action_space.high)
    num_hidden_layers = 0
    neurons_per_hidden_layer = 0

    randomise_hyperparams = True

    if len(sys.argv)==1:

        num_runs = 2000

        for i in range(num_runs):
            print("Evo run: ", i)

            #Generate new domain hyperparams
            if randomise_hyperparams:

                env_kwargs = {
                    'speed_knee' : random.uniform(2., 6.)
                }

                toolbox.register("evaluate", evaluate,
                                 num_inputs=num_inputs, num_outputs=num_outputs,
                                 num_hidden_layers=num_hidden_layers,
                                 neurons_per_hidden_layer=neurons_per_hidden_layer,
                                 render=render, genotype_dir=None, env_kwargs=env_kwargs)

            winner = evo_run(num_inputs, num_outputs, num_hidden_layers,
                             neurons_per_hidden_layer, dir_path, file_name, i,
                             env_kwargs['speed_knee'])
            #Reset strategy
            strategy = cma.Strategy(centroid=centroid, sigma=init_sigma, lambda_=lambda_)
            toolbox.register("generate", strategy.generate, creator.Individual)
            toolbox.register("update", strategy.update)

        #indv_run(winner.get_weights(), num_inputs, num_outputs, num_hidden_layers,
        #         neurons_per_hidden_layer)

    else:

        #Genome directory comes from the command line
        indv_dir = sys.argv[1]

        indv_full_path = dir_path + indv_dir + "/" + file_name

        indv_run(num_inputs, num_outputs, num_hidden_layers,
                 neurons_per_hidden_layer, genotype_dir=indv_full_path)


#Some bug in DEAP means that I have to define toolbox before if __name__ == "__main__"
#apparently

dummy_env = gym.make("BipedalWalker-v3")
state = dummy_env.reset()

num_inputs = len(state)
num_outputs = len(dummy_env.action_space.high)
num_hidden_layers = 0
neurons_per_hidden_layer = 0

render = False

dummy_nn = NeuralNetwork(num_inputs, num_outputs, num_hidden_layers,
                         neurons_per_hidden_layer)
num_weights = dummy_nn.get_num_weights()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate,
                 num_inputs=num_inputs, num_outputs=num_outputs,
                 num_hidden_layers=num_hidden_layers,
                 neurons_per_hidden_layer=neurons_per_hidden_layer,
                 render=render, genotype_dir=None)

#Initial location of distribution centre
centroid = [0.0] * num_weights
#Initial standard deviation of the distribution
init_sigma = 1.0
#Number of children to produce at each generation
#lambda_ = 20 * num_weights
lambda_ = 100
strategy = cma.Strategy(centroid=centroid, sigma=init_sigma, lambda_=lambda_)

toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

if __name__ == "__main__":
    main()
