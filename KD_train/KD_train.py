import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Import Model
from models.DeepECGNet import DeepECGNet


# =============================== Dataset Load ===============================
# Load EEG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/eeg_dataset_ASR_alpha.npz"
data = np.load(loadPath)

x_Train_EEG = data['x_Train']
x_Validate_EEG = data['x_Validate']
x_Test_EEG = data['x_Test']
x_Validate_EEG = data['x_Validate']
y_Train_EEG = data['y_Train']
y_Test_EEG = data['y_Test']
y_Validate_EEG = data['y_Validate']

data.close()

kernels, chans, samples = 1, x_Train_EEG.shape[1], x_Train_EEG.shape[2]

x_Train_EEG = x_Train_EEG.reshape(x_Train_EEG.shape[0], chans, samples, kernels)
x_Validate_EEG = x_Validate_EEG.reshape(x_Validate_EEG.shape[0], chans, samples, kernels)
x_Test_EEG = x_Test_EEG.reshape(x_Test_EEG.shape[0], chans, samples, kernels)

# print("EEG Data Shape")
# print("Train Data Shape : ", x_Train_EEG.shape)         # (5832, 13, 3840, 1)
# print("Test Data Shape : ", x_Test_EEG.shape)           # (766, 13, 3840, 1)
# print("Validate Data Shape : ", y_Validate_EEG.shape)   # (709, 2)
# print("Train Labels Shape : ", y_Train_EEG.shape)       # (5832, 2)
# print("Test Labels Shape : ", y_Test_EEG.shape)         # (766, 2)
# print("Validate Labels Shape : ", y_Validate_EEG.shape) # (709, 2)



# Load ECG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_128_norm.npz"
data = np.load(loadPath)

x_Train_ECG = data['x_Train']
x_Test_ECG = data['x_Test']
x_Validate_ECG = data['x_Validate']
y_Train_ECG = data['y_Train']
y_Test_ECG = data['y_Test']
y_Validate_ECG = data['y_Validate']

data.close()

x_Train_ECG = x_Train_ECG.reshape(x_Train_ECG.shape[0], samples, 1)
x_Validate_ECG = x_Validate_ECG.reshape(x_Validate_ECG.shape[0], samples, 1)
x_Test_ECG = x_Test_ECG.reshape(x_Test_ECG.shape[0], samples, 1)

# print("ECG Data Shape")
# print("Train Data Shape : ", x_Train_ECG.shape)         # (5832, 3840, 1)
# print("Test Data Shape : ", x_Test_ECG.shape)           # (766, 3840, 1)
# print("Validate Data Shape : ", y_Validate_ECG.shape)   # (709, 2)
# print("Train Labels Shape : ", y_Train_ECG.shape)       # (5832, 2)
# print("Test Labels Shape : ", y_Test_ECG.shape)         # (766, 2)
# print("Validate Labels Shape : ", y_Validate_ECG.shape) # (709, 2)



# =============================== Create Model ===============================
# Create Teacher Model : EEGNet
teacher = loaded_model = keras.models.load_model('saved_model/EEGNet_model')
teacher.summary()

loss, acc = loaded_model.evaluate(x_Train_EEG,  y_Train_EEG, verbose=2)
print('Loaded model accuracy : {:5.2f}%'.format(100*acc))
softLabel = loaded_model.predict(x_Train_EEG)
print(softLabel.shape)


# Create Student Model : DeepECGModel
student = DeepECGNet(samples, dropoutRate=0.5)
student.summary()

learnging_rate, epoch = 0.001, 500
optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learnging_rate/epoch, amsgrad=False)
student.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Set Callback
checkpoint_path = "checkpoints/KD/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
# checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1)

logdir="logs/KD/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=100, verbose=0)

student.fit(
    x_Train_ECG,
    softLabel,
    epochs=epoch,
    batch_size=128,
    validation_data=(x_Validate_ECG, y_Validate_ECG),
    callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb]
)

student.save('saved_model/KD_model')

# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD/20210518-191122
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD/20210518-192550  
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD/20210528-210044
