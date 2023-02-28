import tensorflow as tf

x_max = 19.619508366918392
y_max = 27.317051197038307
z_max = 48.05371246231375
x_max = 19.6195
y_max = 27.3171
z_max = 48.0537



@tf.function
def loss_oloop(y_true, y_pred, washout=0):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true[:, washout:, :], y_pred[:, washout:, :])
    return loss


@tf.function
def loss_oloop_l2_reg(y_true, y_pred, washout=0, reg_weight=0.001):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true[washout:, :], y_pred[washout:, :]) + reg_weight * tf.nn.l2_loss(
        y_pred[washout:, :]
    )
    return loss
