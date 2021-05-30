import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load ECG Data numpy format
# loadPath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_128_norm.npz"
loadPath = "/Users/kok_ksy/Desktop/dataset/ecg_dataset_128_norm.npz"
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


# Load Model
ECG_model = keras.models.load_model('saved_model/DeepECGModel_model')

# Model Evaluate
ECG_loss, ECG_acc = ECG_model.evaluate(x_Test,  y_Test, verbose=2)
print('ECG model accuracy : {:5.2f}%'.format(100*ECG_acc))

# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/ECG/20210518-185538

# Load Model
KD_model = keras.models.load_model('saved_model/KD_model')

# Model Evaluate
KD_loss, KD_acc = KD_model.evaluate(x_Test,  y_Test, verbose=2)
print('KD model accuracy : {:5.2f}%'.format(100*KD_acc))

# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/ECG/20210518-192550

