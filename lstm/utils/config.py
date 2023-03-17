from typing import Dict
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
    config["ML_CONSTRAINTS"] = {}
    config["ML_CONSTRAINTS"]["N_EPOCHS"] = args.n_epochs
    config["ML_CONSTRAINTS"]["EPOCH_STEPS"] = args.epoch_steps
    config["ML_CONSTRAINTS"]["BATCH_SIZE"] = args.batch_size
    config["ML_CONSTRAINTS"]["N_CELLS"] = args.n_cells
    config["ML_CONSTRAINTS"]["OLOOP-TRAINED"] = args.oloop_train
    config["ML_CONSTRAINTS"]["OPTIMIZER"] = args.optimizer
    config["ML_CONSTRAINTS"]["ACTIVATION"] = args.activation
    config["ML_CONSTRAINTS"]["LR"] = args.learning_rate
    config["ML_CONSTRAINTS"]["PI WEIGHT"] = args.pi_weighing
    config["ML_CONSTRAINTS"]["DROPOUT"] = args.dropout
    config["ML_CONSTRAINTS"]["REG WEIGHT"] = args.reg_weighing

    # config:: lorenz_data
    config["DATA"] = {}
    config["DATA"]["NORMALISED"] = args.normalised
    config["DATA"]["T_0"] = args.t_0
    config["DATA"]["T_TRANS"] = args.t_trans
    config["DATA"]["T_END"] = args.t_end
    config["DATA"]["DELTA T"] = args.delta_t
    config["DATA"]["TOTAL_N"] = args.total_n
    config["DATA"]["WINDOW_SIZE"] = args.window_size
    config["DATA"]["UPSAMPLING"] = args.upsampling
    config["DATA"]["SIGNAL TO NOISE RATIO"] = args.signal_noise_ratio
    config["DATA"]["TRAINING RATIO"] = args.train_ratio
    config["DATA"]["VALID RATIO"] = args.valid_ratio
    config["DATA"]["STANDARD NORM"] = args.standard_norm

    config["PATHS"] = {}
    config["PATHS"]["lyap_path"] = str(args.lyap_path)
    config["PATHS"]["model_path"] = str(args.model_path)
    config["PATHS"]["data_path"] = str(args.data_path)

    with open(config_path, "w+") as f:
        yaml.dump(config, f)


def generate_config_ks(config_path: Path, args: argparse.Namespace) -> None:
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
    config["ML_CONSTRAINTS"] = {}
    config["ML_CONSTRAINTS"]["DD_LOSS_LABEL"] = args.dd_loss_label
    config["ML_CONSTRAINTS"]["N_EPOCHS"] = args.n_epochs
    config["ML_CONSTRAINTS"]["EPOCH_STEPS"] = args.epoch_steps
    config["ML_CONSTRAINTS"]["BATCH_SIZE"] = args.batch_size
    config["ML_CONSTRAINTS"]["N_CELLS"] = args.n_cells
    config["ML_CONSTRAINTS"]["OLOOP-TRAINED"] = args.oloop_train
    config["ML_CONSTRAINTS"]["OPTIMIZER"] = args.optimizer
    config["ML_CONSTRAINTS"]["ACTIVATION"] = args.activation
    config["ML_CONSTRAINTS"]["LR"] = args.learning_rate
    config["ML_CONSTRAINTS"]["PI WEIGHT"] = args.pi_weighing
    config["ML_CONSTRAINTS"]["DROPOUT"] = args.dropout
    config["ML_CONSTRAINTS"]["REG WEIGHT"] = args.reg_weighing
    config["ML_CONSTRAINTS"]["WASHOUT"] = args.washout

    # config:: lorenz_data
    config["DATA"] = {}
    config["DATA"]["NORMALISED"] = args.normalised
    config["DATA"]["SYS_DIM"] = args.sys_dim
    config["DATA"]["T_0"] = args.t_0
    config["DATA"]["T_TRANS"] = args.t_trans
    config["DATA"]["T_END"] = args.t_end
    config["DATA"]["DELTA T"] = args.delta_t
    config["DATA"]["TOTAL_N"] = args.total_n
    config["DATA"]["WINDOW_SIZE"] = args.window_size
    config["DATA"]["UPSAMPLING"] = args.upsampling
    config["DATA"]["SIGNAL TO NOISE RATIO"] = args.signal_noise_ratio
    config["DATA"]["TRAINING RATIO"] = args.train_ratio
    config["DATA"]["VALID RATIO"] = args.valid_ratio
    config["DATA"]["STANDARD NORM"] = args.standard_norm

    config["KS SOLVER"] = {}
    config["KS SOLVER"]["M"] = args.M
    config["KS SOLVER"]["N"] = args.N
    config["KS SOLVER"]["h"] = args.h
    config["KS SOLVER"]["d"] = args.d

    config["PATHS"] = {}
    config["PATHS"]["lyap_path"] =  str(args.lyap_path)
    config["PATHS"]["model_path"] =  str(args.model_path)
    config["PATHS"]["data_path"] =  str(args.data_path)

    with open(config_path, "w+") as f:
        yaml.dump(config, f)


def load_config_to_dict(model_path: Path) -> Dict:
    """Loads a config file from a given directory and returns it as a dictionary.

    Args:
        model_path (Path): The directory where the configuration file is located.

    Returns:
        A dictionary containing the configuration data.
    """
    config_path = model_path / "config.yml"
    with open(config_path, "r") as stream:
        dict_loaded = yaml.safe_load(stream)
    return dict_loaded


def load_config_to_argparse(model_path: Path) -> argparse.Namespace:
    config_path = model_path / "config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace()
    args.n_epochs = config["ML_CONSTRAINTS"]["N_EPOCHS"]
    args.epoch_steps = config["ML_CONSTRAINTS"]["EPOCH_STEPS"]
    args.batch_size = config["ML_CONSTRAINTS"]["BATCH_SIZE"]
    args.n_cells = config["ML_CONSTRAINTS"]["N_CELLS"]
    args.oloop_train = config["ML_CONSTRAINTS"]["OLOOP-TRAINED"]
    args.optimizer = config["ML_CONSTRAINTS"]["OPTIMIZER"]
    args.activation = config["ML_CONSTRAINTS"]["ACTIVATION"]
    args.learning_rate = config["ML_CONSTRAINTS"]["LR"]
    args.pi_weighing = config["ML_CONSTRAINTS"]["PI WEIGHT"]
    args.dropout = config["ML_CONSTRAINTS"]["DROPOUT"]
    args.reg_weighing = config["ML_CONSTRAINTS"]["REG WEIGHT"]
    args.dd_loss_label = config["ML_CONSTRAINTS"]["DD_LOSS_LABEL"]
    # config:: lorenz_data
    args.normalised = config["DATA"]["NORMALISED"]
    args.t_0 = config["DATA"]["T_0"]
    args.t_trans = config["DATA"]["T_TRANS"]
    args.t_end = config["DATA"]["T_END"]
    args.delta_t = config["DATA"]["DELTA T"]
    args.total_n = config["DATA"]["TOTAL_N"]
    args.window_size = config["DATA"]["WINDOW_SIZE"]
    args.upsampling = config["DATA"]["UPSAMPLING"]
    args.signal_noise_ratio = config["DATA"]["SIGNAL TO NOISE RATIO"]
    args.train_ratio = config["DATA"]["TRAINING RATIO"]
    args.valid_ratio = config["DATA"]["VALID RATIO"]
    return args

    # args.ks_stand = config['DATA']['KS STAND']
    # args.M = config['KS SOLVER']['M']
    # args.N = config['KS SOLVER']['N']
    # args.h = config['KS SOLVER']['h']
    # args.L = config['KS SOLVER']['d']
