import numpy as np
from .data_processing import (
    df_train_valid_test_split,
    train_valid_test_split,
    create_df_nd_random_md_mtm_idx,
    create_test_window,
    create_df_nd_even_md_mtm
)


class Dataclass:
    def __init__(self, model_args) -> None:
        self.model_args = model_args
        if model_args.data_path is None and model_args.lyap_path is None:
            raise ValueError("No reference data provided")
        if model_args.data_path is not None:
            self.data_path = model_args.data_path
            self.load_de_sol()
            self.split_train_valid_test()
        if model_args.lyap_path is not None:
            self.lyap_path = model_args.lyap_path
            self.load_lyap_ref()
            self.t_lyap = model_args.lyap ** (-1)
            self.N_lyap = int(
                self.t_lyap / (model_args.delta_t * model_args.upsampling)
            )

    def load_de_sol(self):
        df = np.genfromtxt(self.data_path, delimiter=",").astype(np.float64)
        self.de_sol = df[1:, :: self.model_args.upsampling]
        self.time = df[0, :: self.model_args.upsampling]
        self.lyap_time = self.time / self.model_args.lyap

    def load_lyap_ref(self):
        self.ref_lyap = np.loadtxt(self.lyap_path)

    def split_train_valid_test(self):
        self.df_train, self.df_valid, self.df_test = df_train_valid_test_split(
            self.de_sol,
            train_ratio=self.model_args.train_ratio,
            valid_ratio=self.model_args.valid_ratio,
        )
        self.time_train, self.time_valid, self.time_test = train_valid_test_split(
            self.time,
            train_ratio=self.model_args.train_ratio,
            valid_ratio=self.model_args.valid_ratio,
        )

        self.idx_lst, self.train_dataset = create_df_nd_random_md_mtm_idx(
            self.df_train.transpose(),
            self.model_args.window_size,
            self.model_args.batch_size,
            self.df_train.shape[0],
            n_random_idx=self.model_args.n_random_idx,
        )
        _, self.valid_dataset = create_df_nd_random_md_mtm_idx(
            self.df_valid.transpose(),
            self.model_args.window_size,
            self.model_args.batch_size,
            1,
            n_random_idx=self.model_args.n_random_idx,
        )
        _, self.test_dataset = create_df_nd_random_md_mtm_idx(
            self.df_test.transpose(),
            self.model_args.window_size,
            self.model_args.batch_size,
            1,
            n_random_idx=self.model_args.n_random_idx,
        )
        for batch, label in self.train_dataset.take(1):
            print(f"Shape of batch: {batch.shape} \n Shape of Label {label.shape}")

    def return_test_window(self, data_type):
        "starting test window"
        if data_type == "train":
            test_window = create_test_window(
                self.df_train, window_size=self.model_args.window_size
            )
        elif data_type == "valid":
            test_window = create_test_window(
                self.df_valid, window_size=self.model_args.window_size
            )
        elif data_type == "test":
            test_window = create_test_window(
                self.df_test, window_size=self.model_args.window_size
            )
        return test_window
