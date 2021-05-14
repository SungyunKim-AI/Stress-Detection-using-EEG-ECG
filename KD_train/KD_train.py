import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import Model
from .model_EEGNet import EEGNet
from .model_ECGModel_v1 import DeepECGModel, ECGModel_v1
from .Distiller import *


# =============================== Dataset Load ===============================
# Load EEG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/eeg_dataset_ASR_alpha_30.npz"
data = np.load(loadPath)

x_Train_eeg = data['x_Train']
x_Test_eeg = data['x_Test']
x_Validate_eeg = data['x_Validate']
y_Train_eeg = data['y_Train']
y_Test_eeg = data['y_Test']
y_Validate_eeg = data['y_Validate']

data.close()

kernels, chans, samples = 1, x_Train_eeg.shape[1], x_Train_eeg.shape[2]

x_Train_eeg = x_Train_eeg.reshape(x_Train_eeg.shape[0], chans, samples, kernels)
x_Validate_eeg = x_Validate_eeg.reshape(x_Validate_eeg.shape[0], chans, samples, kernels)
x_Test_eeg = x_Test_eeg.reshape(x_Test_eeg.shape[0], chans, samples, kernels)

# print("EEG Dataset Shape")
# print("Train Set Shape : ", x_Train_eeg.shape)          # (7528, 13, 3840)
# print("Test Set Shape : ", x_Test_eeg.shape)            # (901, 13, 3840)
# print("Validate Set Shape : ", x_Validate_eeg.shape)    # (972, 13, 3840)
# print("Train Labels Shape : ", y_Train_eeg.shape)       # (7528, 2)
# print("Test Labels Shape : ", y_Test_eeg.shape)         # (901, 2)
# print("Validate Labels Shape : ", y_Validate_eeg.shape) # (972, 2)


# Load ECG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_128_norm.npz"
data = np.load(loadPath)

x_Train_ecg = data['x_Train'][:,2,:]
x_Test_ecg = data['x_Test'][:,2,:]
x_Validate_ecg = data['x_Validate'][:,2,:]
y_Train_ecg = data['y_Train']
y_Test_ecg = data['y_Test']
y_Validate_ecg = data['y_Validate']

data.close()

x_Train_ecg = x_Train_ecg.reshape(x_Train_ecg.shape[0], samples, 1)
x_Validate_ecg = x_Validate_ecg.reshape(x_Validate_ecg.shape[0], samples, 1)
x_Test_ecg = x_Test_ecg.reshape(x_Test_ecg.shape[0], samples, 1)

# print("ECG Dataset Shape")
# print("Train Set Shape : ", x_Train_ecg.shape)          # (6169, 3840, 1)
# print("Test Set Shape : ", x_Test_ecg.shape)            # (771, 3840, 1)
# print("Validate Set Shape : ", x_Validate_ecg.shape)    # (772, 3840, 1)
# print("Train Labels Shape : ", y_Train_ecg.shape)       # (6169, 2)
# print("Test Labels Shape : ", y_Test_ecg.shape)         # (771, 2)
# print("Validate Labels Shape : ", y_Validate_ecg.shape) # (772, 2)



# =============================== Create Model ===============================
# Create EEG Model : Teacher
teacher = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.1, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')
teacher.summary()

# Create ECG Model : Student
student = DeepECGModel(samples, dropout=0.5)
# student = ECGModel_v1(input_dim=(samples,1))
student.summary()


# Train Teacher Model
learnging_rate, epoch = 0.001, 500
optimizer = tf.keras.optimizers.Adam(lr=learnging_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learnging_rate/epoch, amsgrad=False)
teacher.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

checkpoint_path = "checkpoints/EEG/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

logdir="logs/EEG/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, verbose=0)

teacher.fit(
    x_Train_eeg,
    y_Train_eeg,
    epochs=epoch,
    batch_size=128,
    validation_data=(x_Validate_eeg, y_Validate_eeg),
    callbacks= [checkpoint_cb, tensorboard_cb, earlystop_cb]
)

teacher.evaluate(x_Train_eeg[0], y_Test_eeg[0])




# =============================== Distiller ===============================
# # Initialize  distiller
# distiller = Distiller(student=student, teacher=teacher)
# distiller.compile(optimizer=keras.optimizers.Adam(),
#                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
#                  student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                  distillation_loss_fn=keras.losses.KLDivergence(),
#                  alpha=0.3,
#                  temperature=7)

# # Distill teacher to student
# checkpoint_path = "checkpoints/KD/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
# checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

# logdir="logs/KD/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, verbose=0)

# distiller.fit(
#     x_Train_ecg, 
#     y_Train_ecg, 
#     epochs=epoch,
#     callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb]
# )

# # Evaluate student on test dataset
# distiller.evaluate(x_Test_ecg, y_Test_ecg)