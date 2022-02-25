import os
import random

import numpy as np
import tensorflow as tf


def reset_random_seeds():
    os.environ["PYTHONHASHSEED"] = str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)
