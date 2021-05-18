import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from tensorflow.python.keras.backend import softmax

# Import Model
from model_EEGNet import EEGNet
from model_ECGModel_v1 import DeepECGModel, ECGModel_v1
from Distiller import *


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
# Create EEG Model : Teacher
teacher = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.1, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')
teacher.summary()

# Train Teacher Model
learnging_rate, epoch = 0.001, 3
optimizer = keras.optimizers.Adam(lr=learnging_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learnging_rate/epoch, amsgrad=False)
teacher.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

checkpoint_path = "checkpoints/EEG/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
# checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1)

logdir="logs/EEG/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=logdir)

teacher.fit(
    x_Train_EEG,
    y_Train_EEG,
    epochs = epoch,
    batch_size = 128,
    callbacks = [checkpoint_cb, tensorboard_cb]
)


config = tf.ConfigProto()
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
tf.train.Saver().restore(sess, save_path=latest)

i = 0
while i < len(x_Train_EEG):
    j = min(i + batch_size, len(x_Train_EEG))
    batch_xs = x_Train_EEG[i:j,:]
    softmax_values.extend(sess.run(softmax, feed_dict={x: batch_xs, is_training:False}))
    i = j

teacher.load_weights(latest)

# val_loss : 0.51587  val_accuracy : 0.7294
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/EEG/20210517-201022/





# Create ECG Model : Student
student = DeepECGModel(samples, dropout=0.5)
student.summary()

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)


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
    teacher.predict(x_Train_EEG),
    epochs=epoch,
    batch_size=128,
    validation_data=(x_Validate_ECG, y_Validate_ECG),
    callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb]
)

student.evaluate(x_Test_ECG, y_Test_ECG)

# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD/20210517-201710/


# # =============================== Distiller ===============================
# # Initialize  distiller
# distiller = Distiller(student=student, teacher=teacher, train_data=x_Train_EEG)
# distiller.compile(optimizer=keras.optimizers.Adam(),
#                  metrics=[keras.metrics.Accuracy()],
#                  student_loss_fn=keras.losses.BinaryCrossentropy(),
#                  distillation_loss_fn=keras.losses.KLDivergence(),
#                  alpha=0.3,
#                  temperature=7)

# # Distill teacher to student
# checkpoint_path = "checkpoints/KD/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
# checkpoint_cb = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

# logdir="logs/KD/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_cb = keras.callbacks.TensorBoard(log_dir=logdir)

# earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, verbose=0)

# distiller.fit(
#     x_Train_ECG, 
#     y_Train_ECG, 
#     epochs=epoch,
#     callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb]
# )

# # Evaluate student on test dataset
# distiller.evaluate(x_Test_ECG, y_Test_ECG)
