from typing import Union


def decayed_learning_rate(
        step: Union[int, float],
        learning_rate: Union[int, float],
        decay_steps: int = 1000, decay_rate: Union[int, float] = 0.75) -> float:
    """_summary_

    Args:
        step (Union[int, float]): current step
        learning_rate (Union[int, float]): _description_
        decay_steps (int, optional): _description_. Defaults to 1000.
        decay_rate (Union[int, float], optional): _description_. Defaults to 0.75.

    Returns:
        float: current learning rate 
    """
    # careful here! step includes batch steps in the tf framework
    return learning_rate * decay_rate ** (step / decay_steps)
