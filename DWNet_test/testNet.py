# EEGNet-specific imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
import DWNet1D


def testNet(nb_classes, Chans=64, Samples=128,
            dropoutRate=0.2, kernLength=25, F1=8,
            D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

    input1 = Input(shape=(Chans, Samples, 1))
    first_out = DWNet1D.first_level(input1)
    second_out = DWNet1D.second_level(first_out)
    third_out = DWNet1D.third_level(second_out)
    ##################################################################
    # flatten
    flatten = Flatten()(third_out)

    # Neural Network with Dropout
    nn = Dense(units=20,
               activation="relu", name="nn_layer")(flatten)
    do = Dropout(rate=0.1, name="drop")(nn)
    visual = False
    if not visual:
        # Classification Layer
        final_layer = Dense(
            units=2, activation="sigmoid", name="sigmoid_layer")(do)
    else:
        final_layer = do

    return Model(inputs=input1, outputs=final_layer)
