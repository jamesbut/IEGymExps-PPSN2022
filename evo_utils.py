from deap import tools


# This function is just a copy of the function in DEAP but it returns if some
# completion fitness has been met - this prevents unnecessary computation
def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__, completion_fitness=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # Generate a new population
        population = list(toolbox.generate())

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        # End if completion fitness has been achieved
        if completion_fitness is not None:
            if halloffame[0].fitness.values[0] >= completion_fitness:
                print("WINNER FOUND!")
                return population, logbook, True

    print("No winner found :(")

    return population, logbook, False


# This function parses args for '-cmaes_centroid' and then takes the next argument
# after that, which should be an organism directory, and uses that as the centroid.
# This is to start off the search at the point of an already evolved organism.
def get_cmaes_centroid(num_genes, args, dir_path=None, file_name=None):

    import sys
    from neural_network import NeuralNetwork

    if '-cmaes_centroid' in args:

        try:
            org_dir = args[args.index('-cmaes_centroid') + 1]
        except IndexError:
            print('Please provide organism directory after -cmaes_centroid '
                  '(relative to ..../data/)')
            sys.exit(1)

        # Build path of organism
        org_path = dir_path + org_dir + '/' + file_name

        # Read in genotype of organism
        org_nn = NeuralNetwork(genotype_dir=org_path)
        weights = org_nn.get_weights()

        return weights

    else:
        return [0.] * num_genes
