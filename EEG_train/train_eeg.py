import os
import numpy as np
import tensorflow as tf
from datetime import datetime

from utils import plot_loss_curve, plot_acc_curve, normalization

# Import Models
from model_EEGNet import EEGNet
from model_DeepConvNet import DeepConvNet


# Load EEG Data numpy format
# eeg_dataset_ASR_CAR_alpha / eeg_dataset_ASR_CAR_overall / eeg_dataset_CAR_alpha / eeg_dataset_CAR_overall
loadPath = "C:/Users/user/Desktop/numpy_dataset/eeg_dataset_ASR_alpha_30.npz"
data = np.load(loadPath)

x_Train = data['x_Train']
x_Test = data['x_Test']
x_Validate = data['x_Validate']
y_Train = data['y_Train']
y_Test = data['y_Test']
y_Validate = data['y_Validate']

kernels, chans, samples = 1, x_Train.shape[1], x_Train.shape[2]

x_Train = x_Train.reshape(x_Train.shape[0], chans, samples, kernels)
x_Validate = x_Validate.reshape(x_Validate.shape[0], chans, samples, kernels)
x_Test = x_Test.reshape(x_Test.shape[0], chans, samples, kernels)

print("Train Set Shape : ", x_Train.shape)          # (2384, 13, 5120, 1)
print("Test Set Shape : ", x_Test.shape)            # (318, 13, 5120, 1)
print("Validate Set Shape : ", x_Validate.shape)    # (330, 13, 5120, 1)
print("Train Labels Shape : ", y_Train.shape)       # (2868, 2)
print("Test Labels Shape : ", y_Test.shape)         # (394, 2)
print("Validate Labels Shape : ", y_Validate.shape) # (400, 2)

data.close()



###################### model ######################

model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.1, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# model = DeepConvNet(nb_classes = 2, Chans = chans, Samples = samples, dropoutRate = 0.5)



learnging_rate, epoch = 0.001, 500
optimizer = tf.keras.optimizers.Adam(lr=learnging_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learnging_rate/epoch, amsgrad=False)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
model.summary()


# Set Callback
checkpoint_path = "checkpoints/EEG/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

logdir="logs/EEG/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, verbose=0)


fit_model = model.fit(
    x_Train,
    y_Train,
    epochs=epoch,
    batch_size=128,
    validation_data=(x_Validate, y_Validate),
    callbacks= [checkpoint_cb, tensorboard_cb, earlystop_cb]
)


checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
loss, acc = model.evaluate(x_Test,  y_Test, verbose=2)
print("loss : {:5.2f} / accuracy: {:5.2f}%".format(loss, 100*acc))

# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/EEG/20210512-180516



# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/EEG/20210514-022326      # kernLength = 64, F1 = 16, D = 2, F2 = 32,
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/EEG/20210514-023657      # kernLength = 64, F1 = 16, D = 4, F2 = 64,
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/EEG/20210514-024424      # kernLength = 64, F1 =  8, D = 2, F2 = 16,  