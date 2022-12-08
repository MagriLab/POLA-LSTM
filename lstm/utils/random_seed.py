import os
import random
from typing import Optional
import numpy as np
import tensorflow as tf


def reset_random_seeds(random_seed: Optional[int] = 2) -> None:
    """ Reset random seeds in OS, tf, np and random

    Args:
        random_seed (int, optional): random seed to be set globally, Defaults to 2.
    """
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
