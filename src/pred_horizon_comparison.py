
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append('../')
from lstm.closed_loop_tools_mto import prediction_closed_loop
from lstm.lstm_model import build_open_loop_lstm, build_pi_model
from lstm.postprocessing import plots, prediction_horizon
from lstm.preprocessing.data_processing import (create_df_3d,
                                                df_train_valid_test_split,
                                                train_valid_test_split)

tf.keras.backend.set_floatx('float64')

tf.get_logger().setLevel('INFO')
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # noinspection PyPackageRequirements
        import tensorflow as tf
        from tensorflow.python.util import deprecation

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        deprecation.deprecated = deprecated

    except ImportError:
        pass
tensorflow_shutup()

mydf = np.genfromtxt("../src/Lorenz_Data/CSV/10000/Lorenz_trans_001_norm_10000.csv", delimiter=",").astype(np.float64)
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
lt_threshold = [0.05, 0.1, 0.2, 0.5]
model_path = "//Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/euler/10000/"
img_filepath = model_path + "images/time_dev/"
if not os.path.exists(img_filepath):
    os.makedirs(img_filepath)

epochs = np.arange(0, 2001, 1000)
pred_lt = np.zeros((len(lt_threshold), len(epochs)))
pred_lt_l2 = np.zeros((len(lt_threshold), len(epochs)))
kl_div = np.zeros((len(epochs)))

for j in range(len(epochs)):
    model = build_pi_model(10)
    model.load_weights(model_path + "model/" + str(epochs[j]) + "/weights")
    print("Model for ", epochs[j], " epochs successfully loaded.")
    n_length = 700
    lyapunov_time, prediction = prediction_closed_loop(
        model, time_test, df_test, n_length, window_size=window_size
    )
    for i in range(len(lt_threshold)):
        pred_lt_temp = prediction_horizon.predict_horizon_layapunov_time(
            prediction, df_test.T, time_test, window_size=window_size, threshold=lt_threshold[i])
        pred_lt[i, j] = pred_lt_temp
        pred_lt_temp_l2 = prediction_horizon.predict_horizon_layapunov_time_l2(
            prediction, df_test.T, time_test, window_size=window_size, threshold=lt_threshold[i])
        pred_lt_l2[i, j] = pred_lt_temp_l2

    kl_div[j] = prediction_horizon.kl_divergence(prediction, df_test.T, window_size=window_size)

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

print(pred_lt_l2)
plt.title(r"Prediction Horizon for Different Threshold - relative $L_2$")
for i in range(len(lt_threshold)):
    plt.plot(epochs, pred_lt_l2[i, :], '-o', label=str(lt_threshold[i]))
plt.xlabel("Epochs")
plt.ylabel("Prediction Horizon [LT]")
# plt.yscale('log')
plt.legend(loc="center left", bbox_to_anchor=(1.0, 1.0))

plt.savefig(model_path+"images/time_dev/"+"nolog_pred_lt_dev_l2.png", dpi=120, facecolor="w", bbox_inches="tight")
print("Image saved at ", model_path+"images/time_dev/"+"nolog_pred_lt_dev_l2.png")
plt.close()

plt.title("KL Divergence")
plt.plot(epochs, kl_div, '-o')
plt.xlabel("Epochs")
plt.ylabel("KL Divergence")
# plt.yscale('log')
plt.legend(loc="center left", bbox_to_anchor=(1.0, 1.0))

plt.savefig(model_path+"images/time_dev/"+"kl-div.png", dpi=120, facecolor="w", bbox_inches="tight")
print("Image saved at ", model_path+"images/time_dev/"+"kl-div.png")
plt.close()
