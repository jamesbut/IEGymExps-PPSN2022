from deap import tools

#This class extends the DEAP HallOfFame in order to give priority to the YOUNGEST
#winning genotypes. In The Hunting of The Plark, the sparse reward means that all the
#winners have a reward of 1, I therefore think the most recent genotypes will be more
#generalisable and of more interest.
class HallOfFamePriorityYoungest(tools.HallOfFame):

    def update(self, population):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        for ind in population:
            if len(self) == 0 and self.maxsize !=0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if ind.fitness >= self[-1].fitness or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)

"""
This function is just a copy of the function in DEAP but it dumps the best genotype
in the hall of fame every n generations
"""
def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__, dump_every=None, obs_normalise=None,
                     domain_params_in_obs=None, dummy_nn=None, completion_fitness=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    avg_fitnesses = []
    best_fitness_so_far = None
    best_fitnesses = []

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
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

        #Log statistics
        avg_fitnesses.append(record['avg'])

        #Calculate best fitness so far
        if best_fitness_so_far is None:
            best_fitness_so_far = record['max']
        else:
            if record['max'] > best_fitness_so_far:
                best_fitness_so_far = record['max']
        best_fitnesses.append(best_fitness_so_far)

        #End if completion fitness has been achieved
        if completion_fitness is not None:
            if best_fitness_so_far >= completion_fitness:
                print("WINNER FOUND!")
                return population, logbook, avg_fitnesses, best_fitnesses, True

    print("No winner found :(")

    return population, logbook, avg_fitnesses, best_fitnesses, False


import sys

#This function parses args for '-cmaes_centroid' and then takes the next argument
#after that, which should be an organism directory, and uses that as the centroid.
#This is to start off the search at the point of an already evolved organism.
def get_cmaes_centroid(num_genes, args, dir_path=None, file_name=None):

    print(args)

    if '-cmaes_centroid' in args:
        print("HELL YEAH!!")

        try:
            org_dir = args[args.index('-cmaes_centroid')+1]
        except:
            print('Please provide organism directory after -cmaes_centroid '
                  '(relative to ..../data/)')
            sys.exit(1)

        #Build path of organism
        org_path = dir_path + org_dir + '/' + file_name
        print("Org path:", org_path)

        #Read in genotype of organism

    else:
        print("No :(")
        return [0.] * num_genes

