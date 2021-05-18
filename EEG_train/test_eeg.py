import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import Models
from model_EEGNet import EEGNet
from model_DeepConvNet import DeepConvNet


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


# Model Create
model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# model = DeepConvNet(nb_classes = 2, Chans = chans, Samples = samples, dropoutRate = 0.5)

learnging_rate, epoch = 0.001, 500
optimizer = tf.keras.optimizers.Adam(lr=learnging_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learnging_rate/epoch, amsgrad=False)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Model Evaluate
# checkpoint_path = "checkpoints/EEG/20210517-204012/cp-0080.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# latest = tf.train.latest_checkpoint(checkpoint_dir)
loaded_model = tf.keras.models.load_model('saved_model/EEGNet_model')
loss, acc = loaded_model.evaluate(x_Train,  y_Train, verbose=2)
print('복원된 모델의 정확도: {:5.2f}%'.format(100*acc))
pred = loaded_model.predict(x_Train)
print(pred.shape)
# i, batch_size = 0, 128
# x = tf.placeholder(tf.float32, shape=[x_Train.shape[0], chans, samples, kernels], name='Input')
# y = tf.placeholder(tf.float32, shape=[x_Train.shape[0], 2], name='Output')
# z = tf.placeholder(tf.float32, shape=[x_Train.shape[0], 2], name='Output')
# is_training = tf.placeholder_with_default(True, shape=())

# softmax = tf.nn.softmax()
# softmax_values = np.zeros(shape=(len(x_Train), 2), dtype=np.float)

# while i < len(x_Train):
#     j = min(i + batch_size, len(x_Train))
#     batch_xs = x_Train[i:j,:]
#     softmax_values[i:j] = model(softmax, feed_dict={x: batch_xs, is_training:False})
#     i = j

# loss, acc = model.evaluate(x_Test,  y_Test, verbose=2)
# print("loss : {:5.2f} / accuracy: {:5.2f}%".format(loss, 100*acc))