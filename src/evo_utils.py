from deap import tools
from agent import Agent
from env_wrapper import EnvWrapper
from data import dump_list
from typing import Optional
import uuid


# This function is just a copy of the function in DEAP but it returns if some
# completion fitness has been met - this prevents unnecessary computation
def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__, completion_fitness=None,
                     quit_domain_when_complete=True, decoder=None,
                     pop_size=None, p_lb=None, p_ub=None, agent=None,
                     exp_dir_path=None, log_config=None, env_wrapper=None,
                     dump_every=None, analyser=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Use UUID for run name
    run_path = exp_dir_path + str(uuid.uuid4()) + '/'
    winner_found = False

    for gen in range(ngen):
        # Generate a new population
        population = _generate_population(toolbox, decoder, pop_size, p_lb, p_ub)

        # Evaluate the individuals
        results = list(toolbox.map(toolbox.evaluate, population))

        for ind, res in zip(population, results):
            ind.fitness.values = res['fitness']

        # Collect data for analysis
        analyser.collect(results, gen == ngen - 1)

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        # Check whether winner has been found
        if completion_fitness is not None:
            if halloffame[0].fitness.values[0] >= completion_fitness:
                winner_found = True

        # Write evolutionary data to file
        write_evo_data(run_path, log_config, agent, halloffame, logbook, env_wrapper,
                       gen, dump_every, winner_found, complete=False)

        # End if domain is complete
        if quit_domain_when_complete and winner_found:
            break

    # Write evolutionary data to file
    write_evo_data(run_path, log_config, agent, halloffame, logbook, env_wrapper,
                   gen, dump_every, winner_found, complete=True)

    print("WINNER FOUND!") if winner_found else print("No winner found :(")
    return population, logbook, winner_found


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
        raise ValueError('Length of phenotype {' + str(len(phenotypes[0]))
                         + '} and length of phenotype lower bounds {' + str(len(p_lb))
                         + '} are not equal')
    if len(phenotypes[0]) != len(p_ub):
        raise ValueError('Length of phenotype {' + str(len(phenotypes[0]))
                         + '} and length of phenotype upper bounds {' + str(len(p_ub))
                         + '} are not equal')

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


# Write evolutionary run data to file
def write_evo_data(run_path: str, log_config: dict, agent: Agent, hof, logbook,
                   env_wrapper: EnvWrapper, curr_gen: int, dump_every: Optional[int],
                   winner_found: bool, complete: bool):

    # Check whether to dump
    dump: bool = complete
    if dump_every is not None:
        if curr_gen % dump_every == 0:
            dump = True

    if dump:

        # Save if winners and losers are saved OR
        # if only winners are saved and there has been a winner
        if ((log_config['save_winners_only'] is False)
             or (log_config['save_winners_only'] is True and winner_found)):

            # Set dummy agent up to dump
            agent.genotype = hof[0]
            agent.fitness = hof[0].fitness.values[0]
            g_saved = agent.save(run_path, log_config['winner_file_name'],
                                 env_wrapper, log_config['save_if_wb_exceeded'])

            # Save population statistics
            if g_saved:
                dump_list(logbook.select('avg'), run_path, 'mean_fitnesses')
                dump_list(logbook.select('max'), run_path, 'best_fitnesses')
