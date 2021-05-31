import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from sklearn.model_selection import KFold

# Import Model
from models.DeepECGNet import DeepECGNet
from Distiller import Distiller


# =============================== Dataset Load ===============================
# Load EEG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/eeg_dataset_ASR_alpha.npz"
# loadPath = "/Users/kok_ksy/Desktop/dataset/eeg_dataset_ASR_alpha.npz"
data = np.load(loadPath)
x_Train_EEG = data['x_Train']
data.close()

numOfTrainSet, chans, sampleLength =  x_Train_EEG.shape[0], x_Train_EEG.shape[1], x_Train_EEG.shape[2]
x_Train_EEG = x_Train_EEG.reshape(numOfTrainSet, chans, sampleLength, 1)        # (5832, 13, 3840, 1)


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

x_Train_ECG = x_Train_ECG.reshape(x_Train_ECG.shape[0], sampleLength, 1)
x_Validate_ECG = x_Validate_ECG.reshape(x_Validate_ECG.shape[0], sampleLength, 1)
x_Test_ECG = x_Test_ECG.reshape(x_Test_ECG.shape[0], sampleLength, 1)

# print("ECG Data Shape")
# print("Train Data Shape : ", x_Train_ECG.shape)         # (5832, 3840, 1)
# print("Test Data Shape : ", x_Test_ECG.shape)           # (766, 3840, 1)
# print("Validate Data Shape : ", y_Validate_ECG.shape)   # (709, 2)
# print("Train Labels Shape : ", y_Train_ECG.shape)       # (5832, 2)
# print("Test Labels Shape : ", y_Test_ECG.shape)         # (766, 2)
# print("Validate Labels Shape : ", y_Validate_ECG.shape) # (709, 2)



# =============================== Create Model ===============================
# Load Pre-trained(Teacher) Model : EEGNet
teacher = keras.models.load_model('saved_model/EEGNet_model2')
teacher.summary()

# loss, acc = teacher.evaluate(x_Train_EEG,  y_Train_EEG, verbose=2)
# print('Loaded model accuracy : {:5.2f}%'.format(100*acc))
# softLabel = teacher.predict(x_Train_EEG)
# print(softLabel.shape)


# Create Student Model : DeepECGModel
student = DeepECGNet(sampleLength, dropoutRate=0.5)
student.summary()


# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher, chans=chans)

learnging_rate, epoch = 0.001, 5
optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learnging_rate/epoch, amsgrad=False)
distiller.compile(
    optimizer=optimizer,
    student_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
    metrics=['accuracy']
)

# Set Callback
checkpoint_path = "checkpoints/KD/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp-{epoch:04d}.ckpt"
# checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1)

logdir="logs/KD2/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=100, verbose=0)

x_distiller = np.concatenate((x_Train_EEG, x_Train_ECG.reshape(numOfTrainSet,1,sampleLength,1)),axis=1)
distiller.fit(
    x_distiller, 
    y_Train_ECG,
    epochs=epoch,
    batch_size=128,
    validation_data=(x_Validate_ECG, y_Validate_ECG),
    callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb]
)

print("Model Evaluate")
distiller.evaluate(x_Test_ECG, y_Test_ECG)

# distiller.save('saved_model/KD_model2')
distiller.save_weights('./saved_model/KD_model2/checkpoint')

# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD/20210518-191122
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD/20210518-192550  
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD/20210528-210044

# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD2/20210530-182815
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/KD2/20210530-185304
