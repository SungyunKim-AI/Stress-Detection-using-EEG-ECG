import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load EEG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/eeg_dataset_ASR_alpha.npz"
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


# Load Model
loaded_model = keras.models.load_model('saved_model/EEGNet_model')

# Model Evaluate
loss, acc = loaded_model.evaluate(x_Train,  y_Train, verbose=2)
print('Loaded model accuracy : {:5.2f}%'.format(100*acc))

# Softmax values
pred = loaded_model.predict(x_Train)
print(pred.shape)

# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/EEG/20210518-183141
