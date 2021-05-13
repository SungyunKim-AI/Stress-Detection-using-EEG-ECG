import os
import numpy as np
import tensorflow as tf
from datetime import datetime

from tensorflow.python.keras.backend import dropout

from utils import plot_loss_curve, plot_acc_curve, normalization

# Import Models
from model_ECGModel_v1 import *
from model_DeepConvNet import DeepConvNet

# Load ECG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_256.npz"
data = np.load(loadPath)

x_Train = data['x_Train'][:,2,:]
x_Test = data['x_Test'][:,2,:]
x_Validate = data['x_Validate'][:,2,:]
y_Train = data['y_Train']
y_Test = data['y_Test']
y_Validate = data['y_Validate']

data.close()

# 1D
samples = x_Train.shape[1]
x_Train = x_Train.reshape(x_Train.shape[0], samples, 1)
x_Validate = x_Validate.reshape(x_Validate.shape[0], samples, 1)
x_Test = x_Test.reshape(x_Test.shape[0], samples, 1)

# 2D
# chans, samples = x_Train.shape[1], x_Train.shape[2]
# x_Train = x_Train.reshape(x_Train.shape[0], chans, samples, 1)
# x_Validate = x_Validate.reshape(x_Validate.shape[0], chans, samples, 1)
# x_Test = x_Test.reshape(x_Test.shape[0], chans, samples, 1)


print("Train Set Shape : ", x_Train.shape)          # (2384, 13, 5120, 1)
print("Test Set Shape : ", x_Test.shape)            # (318, 13, 5120, 1)
print("Validate Set Shape : ", x_Validate.shape)    # (330, 13, 5120, 1)
print("Train Labels Shape : ", y_Train.shape)       # (2868, 2)
print("Test Labels Shape : ", y_Test.shape)         # (394, 2)
print("Validate Labels Shape : ", y_Validate.shape) 



###################### model ######################

# 1D
model = DeepECGModel(samples, dropout=0.5)
# model = ECGModel_v1(samples)
# model = CustomModel(samples)

# 2D
# model = DeepConvNet(nb_classes=2, Chans=chans, Samples=samples, dropoutRate=0.5)


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# optimizer : health monitor는 SGD, 다른건 adam!!
# model.compile(
#     loss='binary_crossentropy',
#     optimizer=tf.keras.optimizers.SGD(lr=0.01),
#     metrics=['accuracy']
# )

model.summary()

# Set Callback
checkpoint_path = "checkpoints/ECG/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

logdir="logs/ECG/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=0)

decay = 0.001 / 1000      # initial_learning_rate / epochs   (Adam defaults learning_rate = 0.001)
lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * 1 / (1+decay*epoch), verbose=0)


fit_model = model.fit(
    x_Train,
    y_Train,
    epochs=1000,
    batch_size=128,
    validation_data=(x_Validate, y_Validate),
    shuffle=True,
    callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb, lr_decay_cb]
)

checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
loss, acc = model.evaluate(x_Test,  y_Test, verbose=2)
print("loss : {:5.2f} / accuracy: {:5.2f}%".format(loss, 100*acc))


# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/EEG/20210512-180516
