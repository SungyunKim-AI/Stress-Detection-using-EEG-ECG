import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import Model
from models.DeepECGNet import DeepECGNet
from Distiller import Distiller

# Load ECG Data numpy format
loadPath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_128_norm.npz"
# loadPath = "/Users/kok_ksy/Desktop/dataset/ecg_dataset_128_norm.npz"
data = np.load(loadPath)

x_Train = data['x_Train']
x_Test = data['x_Test']
x_Validate = data['x_Validate']
y_Train = data['y_Train']
y_Test = data['y_Test']
y_Validate = data['y_Validate']

data.close()

# 1D
sampleLength = x_Train.shape[1]
x_Train = x_Train.reshape(x_Train.shape[0], sampleLength, 1)
x_Validate = x_Validate.reshape(x_Validate.shape[0], sampleLength, 1)
x_Test = x_Test.reshape(x_Test.shape[0], sampleLength, 1)


# Load Model
ECG_model = keras.models.load_model('saved_model/DeepECGNet_model')

# Model Evaluate
ECG_loss, ECG_acc = ECG_model.evaluate(x_Test,  y_Test, verbose=2)
print('ECG model accuracy : {:5.2f}%'.format(100*ECG_acc))

# Load Tensorboard
# tensorboard --logdir=/Users/user/Desktop/Graduation-Project/logs/ECG/20210518-185538





# KD_model1 = DeepECGNet(sampleLength, dropoutRate=0)

# # checkpoint_path1 = "saved_model/checkpoint"
# KD_model1.load_weights('./saved_model/KD_model2/checkpoint')

# # Model Evaluate
# KD_loss1, KD_acc1 = KD_model1.evaluate(x_Test,  y_Test, verbose=2)
# print('KD model1 accuracy : {:5.2f}%'.format(100*KD_acc1))



# Create Student Model : DeepECGModel
teacher = keras.models.load_model('saved_model/EEGNet_model2')
student = DeepECGNet(sampleLength, dropoutRate=0)
distiller = Distiller(student=student, teacher=teacher, chans=13)

distiller.compile(
    optimizer='adam',
    student_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
    metrics=['accuracy']
)

# 20210530-185304 / 20210530-182815
checkpoint_path = "20210530-185304"
checkpoint_dir = os.path.dirname("checkpoints/KD/" + checkpoint_path + "/cp-{epoch:04d}.ckpt")
latest = tf.train.latest_checkpoint(checkpoint_dir)
distiller.load_weights(latest)

# Model Evaluate
distiller.evaluate(x_Test,  y_Test)
