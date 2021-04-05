import os
import tensorflow as tf
from tensorflow import keras

# Import Models
from model_EEGNet import EEGNet
from model_DeepConvNet import DeepConvNet


kernels, chans, samples = 1, 13, 5120

model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# model = DeepConvNet(nb_classes = 2, Chans = chans, Samples = samples, dropoutRate = 0.5)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# model evaluate
checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
loss, acc = model.evaluate(X_validate,  Y_validate, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))