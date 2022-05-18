import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def generate_config(config_path: Path, args: argparse.Namespace) -> None:
    """Write YAML config file.
    Parameters
    ----------
    config_path: Path
        Path to write YAML config file to.
    args: argparse.Namespace
        Arguments passed from experiment
    """
    # make config directory if it doesn't exist

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config: Dict[str, Dict[str, Any]] = {}

    # config :: ml_constraints
    config['ML_CONSTRAINTS'] = {}
    config['ML_CONSTRAINTS']['N_EPOCHS'] = args.n_epochs
    config['ML_CONSTRAINTS']['EPOCH_STEPS'] = args.epoch_steps
    config['ML_CONSTRAINTS']['EPOCH_ITER'] = args.epoch_iter
    config['ML_CONSTRAINTS']['BATCH_SIZE'] = args.batch_size
    config['ML_CONSTRAINTS']['N_CELLS'] = args.n_cells
    config['ML_CONSTRAINTS']['OLOOP-TRAINED'] = args.oloop_train
    config['ML_CONSTRAINTS']['CLOOP-TRAINED'] = args.cloop_train
    config['ML_CONSTRAINTS']['OPTIMIZER'] = args.optimizer
    config['ML_CONSTRAINTS']['ACTIVATION'] = args.activation
    config['ML_CONSTRAINTS']['LR'] = args.learning_rate
    config['ML_CONSTRAINTS']['L2'] = args.l2_regularisation
    config['ML_CONSTRAINTS']['DROPOUT'] = args.dropout
    config['ML_CONSTRAINTS']['PHYSICS INFORMED'] = args.physics_informed
    config['ML_CONSTRAINTS']['PHYSICS WEIGHT'] = args.physics_weighing

    # config:: lorenz_data
    config['LORENZ_DATA'] = {}
    config['LORENZ_DATA']['NORMALISED'] = args.normalised
    config['LORENZ_DATA']['T_0'] = args.t_0
    config['LORENZ_DATA']['T_TRANS'] = args.t_trans
    config['LORENZ_DATA']['T_END'] = args.t_end
    config['LORENZ_DATA']['DELTA T'] = args.delta_t
    config['LORENZ_DATA']['TOTAL_N'] = args.total_n
    config['LORENZ_DATA']['WINDOW_SIZE'] = args.window_size
    config['LORENZ_DATA']['SIGNAL TO NOISE RATIO'] = args.signal_noise_ratio

    with open(config_path, 'w+') as f:
        yaml.dump(config, f)


def generate_config_sweep(config_path: Path, args: argparse.Namespace) -> None:
    """Write YAML config file.
    Parameters
    ----------
    config_path: Path
        Path to write YAML config file to.
    args: argparse.Namespace
        Arguments passed from experiment
    """
    # make config directory if it doesn't exist

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config: Dict[str, Dict[str, Any]] = {}

    # config :: ml_constraints
    config['ML_CONSTRAINTS'] = {}
    config['ML_CONSTRAINTS']['N_EPOCHS'] = args.n_epochs
    config['ML_CONSTRAINTS']['EPOCH_STEPS'] = args.epoch_steps
    config['ML_CONSTRAINTS']['EPOCH_ITER'] = args.epoch_iter
    config['ML_CONSTRAINTS']['BATCH_SIZE'] = args.batch_size
    config['ML_CONSTRAINTS']['N_CELLS'] = args.n_cells
    config['ML_CONSTRAINTS']['OLOOP-TRAINED'] = args.oloop_train
    config['ML_CONSTRAINTS']['CLOOP-TRAINED'] = args.cloop_train
    config['ML_CONSTRAINTS']['OPTIMIZER'] = args.optimizer
    config['ML_CONSTRAINTS']['ACTIVATION'] = args.activation
    config['ML_CONSTRAINTS']['LR'] = args.learning_rate
    config['ML_CONSTRAINTS']['L2'] = args.l2_regularisation
    config['ML_CONSTRAINTS']['DROPOUT'] = args.dropout
    config['ML_CONSTRAINTS']['PHYSICS INFORMED'] = args.physics_informed
    config['ML_CONSTRAINTS']['PHYSICS WEIGHT'] = args.physics_weighing

    # config:: lorenz_data
    config['LORENZ_DATA'] = {}
    config['LORENZ_DATA']['NORMALISED'] = args.normalised
    config['LORENZ_DATA']['T_0'] = args.t_0
    config['LORENZ_DATA']['T_TRANS'] = args.t_trans
    config['LORENZ_DATA']['T_END'] = args.t_end
    config['LORENZ_DATA']['DELTA T'] = args.delta_t
    config['LORENZ_DATA']['TOTAL_N'] = args.total_n
    config['LORENZ_DATA']['WINDOW_SIZE'] = args.window_size
    config['LORENZ_DATA']['SIGNAL TO NOISE RATIO'] = args.signal_noise_ratio

    with open(config_path, 'w+') as f:
        yaml.dump(config, f)
