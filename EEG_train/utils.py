import numpy as np
from matplotlib import pyplot as plt

# Loss Graph
def plot_loss_curve(history):
    plt.figure(figsize=(15, 10))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


# Accuracy Graph
def plot_acc_curve(history):
    plt.figure(figsize=(15, 10))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


# Normalization
def normalization(X, hypes):
    print("Normalizing...")
    if hypes == "01":
        x_min = np.amin(X)
        x_max = np.amax(X)
        X = (X - x_min)/(x_max-x_min)
    elif hypes == "-11":
        std = np.std(np.array(X).ravel())
        mean = np.mean(np.array(X).ravel())
        X = (X - mean)/std
    return X
