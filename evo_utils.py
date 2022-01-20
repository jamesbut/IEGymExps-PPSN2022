from deap import tools


# This function is just a copy of the function in DEAP but it returns if some
# completion fitness has been met - this prevents unnecessary computation
def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__, completion_fitness=None,
                     quit_domain_when_complete=True):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    complete = False

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

        # Check whether domain is complete
        if completion_fitness is not None:
            if halloffame[0].fitness.values[0] >= completion_fitness:
                complete = True
                # End if domain is complete
                if quit_domain_when_complete:
                    print("WINNER FOUND!")
                    return population, logbook, True

    print("WINNER FOUND!") if complete else print("No winner found :(")
    return population, logbook, complete


# This function determines the starting centroid of the CMAES algorithm.
# It can be used to start off the search at the point of an already evolved organism.
def get_cmaes_centroid(num_genes, json, dir_path=None, file_name=None):

    # If no centroid is given, set it to be 0.0
    if 'centroid' not in json:
        return [0.0] * num_genes

    else:
        # Read in centroid
        centroid_json = json['centroid']

        # If centroid is string, read in genome from file path
        if isinstance(centroid_json, str):

            import sys
            from agent import Agent

            # Build organism path
            org_path = dir_path + centroid_json + '/' + file_name

            # Read in genotype of organism
            org = Agent(agent_path=org_path)
            return org.genotype

        # If centroid is number, return number as vector
        else:
            return [centroid_json] * num_genes
