import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sys.path.append('../')
from lstm.closed_loop_tools import (add_new_pred, compute_lyapunov_time_arr,
                                    create_window_closed_loop,
                                    prediction_closed_loop)
from lstm.loss import backward_diff, loss_oloop, pi_loss
from lstm.lstm_model import (build_open_loop_lstm, build_pi_model,
                             load_open_loop_lstm)
from lstm.postprocessing import plots, prediction_horizon
from lstm.preprocessing.data_processing import (create_df_3d,
                                                create_training_split,
                                                df_train_valid_test_split,
                                                df_training_split,
                                                train_valid_test_split)
from lstm.utils.config import generate_config
from lstm.utils.random_seed import reset_random_seeds

tf.get_logger().setLevel('INFO')
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


mydf = np.genfromtxt("../src/Lorenz_Data/CSV/100000/Lorenz_trans_001_norm_100000.csv", delimiter=",")
time = mydf[0, :]
mydf = mydf[1:, :]
df_train, df_valid, df_test = df_train_valid_test_split(mydf)
time_train, time_valid, time_test = train_valid_test_split(time)
x_train, x_valid, x_test = train_valid_test_split(mydf[0, :])
y_train, y_valid, y_test = train_valid_test_split(mydf[1, :])
z_train, z_valid, z_test = train_valid_test_split(mydf[2, :])

# Windowing
window_size = 100
batch_size = 32
cells = 10
shuffle_buffer_size = df_train.shape[0]
train_dataset = create_df_3d(
    df_train.transpose(), window_size, batch_size, shuffle_buffer_size
)
valid_dataset = create_df_3d(df_valid.transpose(), window_size, batch_size, 1)
test_dataset = create_df_3d(df_test.transpose(), window_size, batch_size, 1)
lt_threshold = [0.05, 0.1, 0.2]
model_path = "//Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/100000/"
img_filepath = model_path + "images/time_dev/"
if not os.path.exists(img_filepath):
    os.makedirs(img_filepath)

epochs = np.arange(10, 500, 10)
pred_lt = np.zeros((len(lt_threshold), len(epochs)))

for j in range(len(epochs)):
    model = build_pi_model(10)
    model.load_weights(model_path + "model/" + str(epochs[j]) + "/weights")
    n_length = 700
    lyapunov_time, prediction = prediction_closed_loop(
        model, time_test, df_test, n_length, window_size=window_size
    )
    for i in range(len(lt_threshold)):
        pred_lt_temp = prediction_horizon.predict_horizon_layapunov_time(
            prediction, df_test.T, time_test, window_size=100, threshold=lt_threshold[i])
        pred_lt[i, j] = pred_lt_temp

    # fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, facecolor="white")  # , figsize=(15, 14))
    # axs[0].plot(lyapunov_time[:n_length], df_test.T[window_size: window_size+n_length, 0], 'k')
    # axs[0].plot(lyapunov_time[:n_length], prediction[:n_length, 0], 'r--')
    # axs[1].plot(lyapunov_time[:n_length], df_test.T[window_size: window_size+n_length, 1], 'k')
    # axs[1].plot(lyapunov_time[:n_length], prediction[:n_length, 1], 'r--')
    # axs[2].plot(lyapunov_time[:n_length], df_test.T[window_size: window_size+n_length, 2], 'k', label="Numerical Solution")
    # axs[2].plot(lyapunov_time[:n_length], prediction[:n_length, 2], 'r--', label="Prediction")
    # axs[2].legend(loc="center left", bbox_to_anchor=(1.0, 2.0))
    # axs[2].set_ylim(-1,1)
    # fig.suptitle("Epoch %d Prediction Horizon threshold: %.1f - LT: %4f" % (j, lt_threshold[-1], pred_lt_temp))
    # fig.savefig(img_filepath+str(j)+ ".png", dpi=200, facecolor="w", bbox_inches="tight")
    # print("prediction saved at ", img_filepath+str(j)+ ".png")
    # plt.close(fig)

print(pred_lt)
plt.title("Prediction Horizon for Different Threshold")
for i in range(len(lt_threshold)):
    plt.plot(epochs, pred_lt[i, :], '-o', label=str(lt_threshold[i]))
plt.xlabel("Epochs")
plt.ylabel("Prediction Horizon [LT]")
# plt.yscale('log')
plt.legend(loc="center left", bbox_to_anchor=(1.0, 1.0))

plt.savefig(model_path+"images/time_dev/"+"nolog_pred_lt_dev.png", dpi=120, facecolor="w", bbox_inches="tight")
print("Image saved at ", model_path+"images/time_dev/"+"nolog_pred_lt_dev.png")
plt.close()
