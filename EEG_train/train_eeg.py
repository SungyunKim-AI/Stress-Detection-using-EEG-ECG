import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from utils import plot_loss_curve, plot_acc_curve, normalization
from Load_Data import Load_Data

# Import Models
from model_EEGNet import EEGNet
from model_DeepConvNet import DeepConvNet


# # Load EEG Data
# EEG, Labels, numOfBaseline, numOfStimuli, samples = load_eeg_data()

# kernels, chans = 1, 13

# # Train : Validate : Test = 7 : 1.5 : 1.5
# X_train, X_validate, Y_train, Y_validate = train_test_split(
#     EEG, Labels, test_size=0.3, random_state=42)
# X_validate, X_test, Y_validate, Y_test = train_test_split(
#     X_validate, Y_validate, test_size=0.5, random_state=42)

# # Load ECG Data_2
[X_train, X_test, X_validate, Y_train, Y_test, Y_validate] = Load_Data().load_eeg_data_2()
kernels, chans, samples = 1, 13, X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print("Train Set Shape : ", X_train.shape)          # (2868, 13, 5120, 1)
print("Test Set Shape : ", X_test.shape)            # (394, 13, 5120, 1)
print("Validate Set Shape : ", X_validate.shape)    # (400, 13, 5120, 1)
print("Train Labels Shape : ", Y_train.shape)       # (2868, 2)
print("Test Labels Shape : ", Y_test.shape)         # (394, 2)
print("Validate Labels Shape : ", Y_validate.shape) # (400, 2)


###################### model ######################

model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# model = DeepConvNet(nb_classes = 2, Chans = chans, Samples = samples, dropoutRate = 0.5)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)

# `checkpoint_path` 포맷을 사용하는 가중치를 저장합니다
model.save_weights(checkpoint_path.format(epoch=0))

fit_model = model.fit(
    X_train,
    Y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_validate, Y_validate),
    callbacks=[cp_callback])

plot_loss_curve(fit_model.history)
plot_acc_curve(fit_model.history)
