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
    config['ML_CONSTRAINTS']['BATCH_SIZE'] = args.batch_size
    config['ML_CONSTRAINTS']['N_CELLS'] = args.n_cells
    config['ML_CONSTRAINTS']['OLOOP-TRAINED'] = args.oloop_train
    config['ML_CONSTRAINTS']['OPTIMIZER'] = args.optimizer
    config['ML_CONSTRAINTS']['ACTIVATION'] = args.activation
    config['ML_CONSTRAINTS']['LR'] = args.learning_rate
    config['ML_CONSTRAINTS']['L2'] = args.l2_regularisation
    config['ML_CONSTRAINTS']['DROPOUT'] = args.dropout
    config['ML_CONSTRAINTS']['PHYSICS WEIGHT'] = args.physics_weighing

    # config:: lorenz_data
    config['DATA'] = {}
    config['DATA']['NORMALISED'] = args.normalised
    config['DATA']['T_0'] = args.t_0
    config['DATA']['T_TRANS'] = args.t_trans
    config['DATA']['T_END'] = args.t_end
    config['DATA']['DELTA T'] = args.delta_t
    config['DATA']['TOTAL_N'] = args.total_n
    config['DATA']['WINDOW_SIZE'] = args.window_size
    config['DATA']['UPSAMPLING'] = args.upsampling
    config['DATA']['SIGNAL TO NOISE RATIO'] = args.signal_noise_ratio
    config['DATA']['TRAINING RATIO'] = args.train_ratio
    config['DATA']['VALID RATIO'] = args.valid_ratio
    with open(config_path, 'w+') as f:
        yaml.dump(config, f)

def load_config_to_dict(model_path):
    with open(model_path+"config.yml", 'r') as stream:
        dict_loaded = yaml.safe_load(stream)
    return dict_loaded