'''
Evolutionary algorithm constants
'''

#Initial standard deviation of the CMAES distribution
INIT_SIGMA = 1.

#Number of children to produce at each generation
LAMBDA = 100
NUM_GENS = 100
NUM_EVO_RUNS = 1

#Gene bounds
G_LB = [-10., -10.]
G_UB = [10., 120.]


'''
Evolution execution constants
'''

PARALLELISE = True
RENDER = False
RANDOMISE_HYPERPARAMETERS = False


'''
Environment constants
'''

'''
ENV_NAME = 'BipedalWalker-v3'
DOMAIN_PARAMETERS = [4.75]

ENV_NAME = 'HalfCheetah-v2'
ENV_NAME = 'LunarLanderContinuous-v2'
ENV_NAME = 'HumanoidPyBulletEnv-v0'
ENV_NAME = 'HalfCheetahPyBulletEnv-v0'
ENV_NAME = 'InvertedDoublePendulum-v2'
'''

ENV_NAME = 'MountainCarContinuous-v0'
COMPLETION_FITNESS = 2.15
#Engine powers
#DOMAIN_PARAMETERS = [0.0010]
DOMAIN_PARAMETERS = [0.0010, 0.001001]
#DOMAIN_PARAMETERS = [0.0008, 0.0012, 0.0016]

#COMPLETION_FITNESS = None

if 'PyBulletEnv' in ENV_NAME:
    import pybulletgym


'''
Controller network constants
'''

NUM_HIDDEN_LAYERS = 0
NEURONS_PER_HIDDEN_LAYER = 0
BIAS = False

#Evolve solution using indirect encoding
USE_DECODER = False

#Weight bounds
W_LB = [-10., -10.]
W_UB = [10., 120.]

#Enforce the weight bounds
#If this is turned off the weight bounds are not applied
ENFORCE_WB = False



'''
Logging constants
'''

DATA_DIR_PATH = "../IndirectEncodingsExperiments/lib/NeuroEvo/data/"
WINNER_FILE_NAME = "best_winner_so_far"

#If this is turned off the genome is not saved if weight bounds are exceeded
SAVE_IF_WB_EXCEEDED = True
#Only save solutions that have exceeded the completion fitness
SAVE_WINNERS_ONLY = False
