import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Import Models
from model_ECG import ECGModel_v1, DeepECGModel

# Load ECG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_128_norm.npz"
data = np.load(loadPath)

x_Train = data['x_Train']
x_Test = data['x_Test']
x_Validate = data['x_Validate']
y_Train = data['y_Train']
y_Test = data['y_Test']
y_Validate = data['y_Validate']

data.close()


# 1D
samples = x_Train.shape[1]
x_Train = x_Train.reshape(x_Train.shape[0], samples, 1)
x_Validate = x_Validate.reshape(x_Validate.shape[0], samples, 1)
x_Test = x_Test.reshape(x_Test.shape[0], samples, 1)

print("Train Set Shape : ", x_Train.shape)          # (5832, 3840, 1)
print("Test Set Shape : ", x_Test.shape)            # (766, 3840, 1)
print("Validate Set Shape : ", x_Validate.shape)    # (709, 3840, 1)
print("Train Labels Shape : ", y_Train.shape)       # (5832, 2)
print("Test Labels Shape : ", y_Test.shape)         # (766, 2)
print("Validate Labels Shape : ", y_Validate.shape) # (709, 2)




###################### model ######################
model = DeepECGModel(samples, dropout=0.1)

learnging_rate, epoch = 0.001, 500
optimizer = keras.optimizers.Adam(lr=learnging_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learnging_rate/epoch, amsgrad=False)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.summary()

# Set Callback
checkpoint_path = "checkpoints/ECG/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1)

logdir="logs/ECG/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=logdir)

earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=100, verbose=0)

model.fit(
    x_Train,
    y_Train,
    epochs=epoch,
    batch_size=128,
    validation_data=(x_Validate, y_Validate),
    callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb]
)

model.save('saved_model/DeepECGModel_model')

# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/ECG/20210518-185538
