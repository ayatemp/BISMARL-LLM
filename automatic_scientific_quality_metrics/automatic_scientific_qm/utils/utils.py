import random
import numpy as np
import torch


def seeding(seed):
    """
    Seed all random number generators.

    Args:
        seed (int): The seed to use for seeding the random number generators.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
