from deap import tools


# This function is just a copy of the function in DEAP but it returns if some
# completion fitness has been met - this prevents unnecessary computation
def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__, completion_fitness=None,
                     quit_domain_when_complete=True, decoder=None,
                     pop_size=None, p_lb=None, p_ub=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    complete = False

    for gen in range(ngen):
        # Generate a new population
        population = _generate_population(toolbox, decoder, pop_size, p_lb, p_ub)

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


# Generates new population using evolutionary algorithm
def _generate_population(toolbox, decoder, pop_size, p_lb, p_ub):

    # If evolution is not using an indirect encoding or phenotype bounds are not given
    if decoder is None or p_lb is None or p_ub is None:
        return list(toolbox.generate())
    # If an IE is used and phenotype bounds are given, we have to see whether the
    # phenotypes are within the phenotype bounds.
    else:

        population = []
        num_remain_slots = pop_size
        count = 0

        while num_remain_slots > 0:

            # Propose population
            proposed_genos = list(toolbox.generate(num_indvs=num_remain_slots))

            # Push genotypes through indirect encoding
            proposed_phenos = list(map(
                lambda geno: decoder.forward(geno).detach().numpy(),
                proposed_genos))

            # Check which of the proposed population fits within the phenotype bounds
            bounded_pheno_genos = _filter_genos_by_pheno_bounds(
                proposed_genos, proposed_phenos, p_lb, p_ub)

            # Genotypes of the bounded phenotypes
            population += bounded_pheno_genos

            # Update counting variables
            num_remain_slots = pop_size - len(population)
            count += 1

            # Error message
            if count > 100:
                print('Looping trying to find phenotypes in bounds...')

        return population


def _filter_genos_by_pheno_bounds(genotypes, phenotypes, p_lb, p_ub):

    # Check length are correct
    if len(phenotypes[0]) != len(p_lb):
        raise ValueError('Length of phenotype {' + len(phenotypes[0]) + '} and length '
                         'of phenotype lower bounds {' + len(p_lb) + '} are not equal')
    if len(phenotypes[0]) != len(p_ub):
        raise ValueError('Length of phenotype {' + len(phenotypes[0]) + '} and length '
                         'of phenotype upper bounds {' + len(p_ub) + '} are not equal')

    # Check bounds and remove if phenotype is not within bounds
    bounded_pheno_genos = []
    for geno, pheno in zip(genotypes, phenotypes):
        if _pheno_in_bounds(pheno, p_lb, p_ub):
            bounded_pheno_genos.append(geno)

    return bounded_pheno_genos


# Check with phenotype is in bounds
def _pheno_in_bounds(pheno, p_lb, p_ub) -> bool:

    for (p, lb, ub) in zip(pheno, p_lb, p_ub):
        if p < lb:
            return False
        if p > ub:
            return False

    return True


# Expand bound
def expand_bound(bound, bound_size):
    if bound is not None and len(bound) == 1:
        bound *= bound_size
    return bound


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

            from agent import Agent

            # Build organism path
            org_path = dir_path + centroid_json + '/' + file_name + '.json'

            # Read in genotype of organism
            org = Agent(agent_path=org_path)
            return org.genotype

        # If centroid is number, return number as vector
        else:
            return centroid_json * num_genes
