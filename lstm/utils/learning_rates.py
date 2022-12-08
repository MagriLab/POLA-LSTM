from typing import Union
import math


def decayed_learning_rate(
        step: Union[int, float],
        learning_rate: Union[int, float],
        decay_steps: int = 1000, decay_rate: Union[int, float] = 0.75) -> float:
    """_summary_

    Args:
        step (Union[int, float]): current step
        learning_rate (Union[int, float]): original learning rate
        decay_steps (int, optional): the number of steps to take before applying decay. Defaults to 1000.
        decay_rate (Union[int, float], optional): the number of steps to take before applying decay. Defaults to 0.75.

    Returns:
        float: current learning rate 
    """
    if step <= 0:
            raise ValueError("'step' must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("'learning_rate' must be a positive float")
    if decay_steps <= 0:
        raise ValueError("'decay_steps' must be a positive integer")
    if decay_rate <= 0 or decay_rate > 1:
        raise ValueError("'decay_rate' must be a float between 0 and 1")

    return learning_rate * math.pow(decay_rate, step / decay_steps)
