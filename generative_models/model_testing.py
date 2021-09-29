#A few functions for generating inputs for testing the generative models

import numpy as np
import torch

def code_in_range(code_size, lb, ub, step_size=0.001):

    if code_size == 1:
        code = np.arange(lb, ub, step_size)
        code = code.reshape(len(code), code_size)

    elif code_size == 2:
        code = []
        code_range = np.arange(lb, ub, step_size)
        for i in range(len(code_range)):
            for j in range(len(code_range)):
                code.append([code_range[i], code_range[j]])
        code = np.array(code)

    return torch.from_numpy(code).float()
