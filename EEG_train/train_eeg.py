import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Import Models
from .model_EEGNet import EEGNet
from .model_DeepConvNet import DeepConvNet


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



learnging_rate, epoch = 0.001, 300
optimizer = tf.keras.optimizers.Adam(lr=learnging_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learnging_rate/epoch, amsgrad=False)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
model.summary()


# Set Callback
checkpoint_path = "checkpoints/EEG/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1)

logdir="logs/EEG/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=1e-3, patience=50, verbose=0)


model.fit(
    x_Train,
    y_Train,
    epochs=epoch,
    batch_size=128,
    validation_data=(x_Validate, y_Validate),
    callbacks= [checkpoint_cb, tensorboard_cb, earlystop_cb]
)

model.save('saved_model/EEGNet_model')


# Load Tensorboard

# kernLength = 64, F1 =  8, D = 2, F2 = 16
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/EEG/20210517-204012/cp-0080      # val_loss: 0.4869 - val_accuracy: 0.7715