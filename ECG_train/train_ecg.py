import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from utils import plot_loss_curve, plot_acc_curve, normalization

# Import Models
from model_ECGModel_v1 import ECGModel_v1, DeepECGModel
from model_DeepConvNet import DeepConvNet

# Load ECG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset.npz"
#loadPath = "/Users/kok_ksy/Desktop/dataset/ecg_dataset.npz"
data = np.load(loadPath)

x_Train = data['x_Train']
x_Test = data['x_Test']
x_Validate = data['x_Validate']
y_Train = data['y_Train']
y_Test = data['y_Test']
y_Validate = data['y_Validate']

kernels, chans, samples = 1, x_Train.shape[1], x_Train.shape[2]


# 1D
# x_Train = x_Train[:, 1, :]
# x_Validate = x_Validate[:, 1, :]
# x_Test = x_Test[:, 1, :]
# x_Train = x_Train.reshape(x_Train.shape[0], samples, 1)
# x_Validate = x_Validate.reshape(x_Validate.shape[0], samples, 1)
# x_Test = x_Test.reshape(x_Test.shape[0], samples, 1)
# Train Set Shape :  (1175, 5120, 1)
# Test Set Shape :  (170, 5120, 1)    
# Validate Set Shape :  (167, 5120, 1)
# Train Labels Shape :  (1175, 2)     
# Test Labels Shape :  (170, 2)       
# Validate Labels Shape :  (167, 2) 

# 2D
x_Train = x_Train.reshape(x_Train.shape[0], chans, samples, kernels)
x_Validate = x_Validate.reshape(x_Validate.shape[0], chans, samples, kernels)
x_Test = x_Test.reshape(x_Test.shape[0], chans, samples, kernels)
# Train Set Shape :  (1175, 3, 5120, 1)
# Test Set Shape :  (170, 3, 5120, 1)
# Validate Set Shape :  (167, 3, 5120, 1)
# Train Labels Shape :  (1175, 2)
# Test Labels Shape :  (170, 2)
# Validate Labels Shape :  (167, 2)

print("Train Set Shape : ", x_Train.shape)          # (2384, 13, 5120, 1)
print("Test Set Shape : ", x_Test.shape)            # (318, 13, 5120, 1)
print("Validate Set Shape : ", x_Validate.shape)    # (330, 13, 5120, 1)
print("Train Labels Shape : ", y_Train.shape)       # (2868, 2)
print("Test Labels Shape : ", y_Test.shape)         # (394, 2)
print("Validate Labels Shape : ", y_Validate.shape) 

data.close()



###################### model ######################

# model = ECGModel_v1(samples)
model = DeepConvNet(nb_classes=2, Chans=3, Samples=samples, dropoutRate=0.5)
# model = DeepECGModel(samples)


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

checkpoint_path = "ecg_training_1/cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)

# `checkpoint_path` 포맷을 사용하는 가중치를 저장합니다
model.save_weights(checkpoint_path.format(epoch=0))

fit_model = model.fit(
    x_Train,
    y_Train,
    epochs=300,
    batch_size=16,
    validation_data=(x_Validate, y_Validate),
    callbacks=[cp_callback]
)

plot_loss_curve(fit_model.history)
plot_acc_curve(fit_model.history)
