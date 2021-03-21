#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:33:52 2019
@author: eleftherios
"""
from keras.layers import Conv1D, BatchNormalization, Activation, concatenate
from keras.layers import MaxPooling1D, Flatten, Dense, Dropout
from keras import regularizers


def first_level(input_layer):
    input_layer = input_layer[0]
    index = 0
    _filters = 15
    _padding = "same"
    _kernel_regularizer = 0.001
    _activation = "relu"
    _batch_normalization = True
    c_03 = Conv1D(
        filters=_filters,
        kernel_size=3,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_three_"+str(index))(input_layer)
    c_05 = Conv1D(
        filters=_filters,
        kernel_size=5,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_five_"+str(index))(input_layer)
    c_10 = Conv1D(
        filters=_filters,
        kernel_size=10,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_ten_"+str(index))(input_layer)
    c_20 = Conv1D(
        filters=_filters,
        kernel_size=20,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_twenty_"+str(index))(input_layer)
    c_50 = Conv1D(
        filters=_filters,
        kernel_size=50,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_fifty_"+str(index))(input_layer)
    c_100 = Conv1D(
        filters=_filters,
        kernel_size=100,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_100_"+str(index))(input_layer)
    c_200 = Conv1D(
        filters=_filters,
        kernel_size=200,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_200_"+str(index))(input_layer)

    concatenation_layer = concatenate(
        [c_03, c_05, c_10, c_20, c_50, c_100, c_200], axis=-1)
    print("after: ", concatenation_layer.shape)

    max_pool = MaxPooling1D(pool_size=2)(concatenation_layer)

    if _batch_normalization == True:
        bn = BatchNormalization(name="bn_"+str(index))(max_pool)
        output = Activation(
            _activation, name="first_layer_out")(bn)
    else:
        output = Activation(_activation, name="first_layer_out")(
            concatenation_layer)

    return output


def second_level(input_layer):
    print("second: ", input_layer)
    index = 1
    _filters = 30
    _padding = "same"
    _kernel_regularizer = 0.001
    _activation = "relu"
    _batch_normalization = True
    c_03 = Conv1D(
        filters=_filters,
        kernel_size=3,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_three_"+str(index))(input_layer)
    c_05 = Conv1D(
        filters=_filters,
        kernel_size=5,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_five_"+str(index))(input_layer)
    c_10 = Conv1D(
        filters=_filters,
        kernel_size=10,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_ten_"+str(index))(input_layer)
    c_20 = Conv1D(
        filters=_filters,
        kernel_size=20,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_twenty_"+str(index))(input_layer)
    c_50 = Conv1D(
        filters=_filters,
        kernel_size=50,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_fifty_"+str(index))(input_layer)
    c_100 = Conv1D(
        filters=_filters,
        kernel_size=100,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_100_"+str(index))(input_layer)

    concatenation_layer = concatenate(
        [c_03, c_05, c_10, c_20, c_50, c_100], axis=-1)
    max_pool = MaxPooling1D(pool_size=2)(concatenation_layer)

    if _batch_normalization == True:
        bn = BatchNormalization(name="bn_"+str(index))(max_pool)
        output = Activation(
            _activation, name="second_layer_out")(bn)
    else:
        output = Activation(_activation, name="second_layer_out")(
            concatenation_layer)

    return output


def third_level(input_layer):
    index = 2
    _filters = 60
    _padding = "same"
    _kernel_regularizer = 0.001
    _activation = "relu"
    _batch_normalization = True
    c_03 = Conv1D(
        filters=_filters,
        kernel_size=3,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_three_"+str(index))(input_layer)
    c_05 = Conv1D(
        filters=_filters,
        kernel_size=5,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_five_"+str(index))(input_layer)
    c_10 = Conv1D(
        filters=_filters,
        kernel_size=10,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_ten_"+str(index))(input_layer)
    c_20 = Conv1D(
        filters=_filters,
        kernel_size=20,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_twenty_"+str(index))(input_layer)
    c_50 = Conv1D(
        filters=_filters,
        kernel_size=50,
        strides=1,
        padding=_padding,
        kernel_regularizer=regularizers.l2(
            _kernel_regularizer),
        name="conv_fifty_"+str(index))(input_layer)

    concatenation_layer = concatenate([c_03, c_05, c_10, c_20, c_50], axis=-1)
    max_pool = MaxPooling1D(pool_size=2)(concatenation_layer)

    if _batch_normalization == True:
        bn = BatchNormalization(name="bn_"+str(index))(max_pool)
        output = Activation(
            _activation, name="third_layer_out")(bn)
    else:
        output = Activation(_activation, name="third_layer_out")(
            concatenation_layer)

    return output


# def fourth_level(input_layer, hyperparameters):
#     index = 3
#     c_03 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=3,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_three_"+str(index))(input_layer)
#     c_05 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=5,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_five_"+str(index))(input_layer)
#     c_10 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=10,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_ten_"+str(index))(input_layer)
#     c_20 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=20,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_twenty_"+str(index))(input_layer)

#     concatenation_layer = concatenate([c_03, c_05, c_10, c_20], axis=-1)
#     max_pool = MaxPooling1D(pool_size=2)(concatenation_layer)

#     if hyperparameters["batch_normalization"]:
#         bn = BatchNormalization(name="bn_"+str(index))(max_pool)
#         output = Activation(
#             hyperparameters["activation"], name="fourth_layer_out")(bn)
#     else:
#         output = Activation(hyperparameters["activation"], name="fourth_layer_out")(
#             concatenation_layer)

#     return output


# def fifth_level(input_layer, hyperparameters):
#     index = 4
#     c_03 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=3,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_three_"+str(index))(input_layer)
#     c_05 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=5,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_five_"+str(index))(input_layer)
#     c_10 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=10,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_ten_"+str(index))(input_layer)

#     concatenation_layer = concatenate([c_03, c_05, c_10], axis=-1)
#     max_pool = MaxPooling1D(pool_size=2)(concatenation_layer)

#     if hyperparameters["batch_normalization"]:
#         bn = BatchNormalization(name="bn_"+str(index))(max_pool)
#         output = Activation(
#             hyperparameters["activation"], name="fifth_layer_out")(bn)
#     else:
#         output = Activation(hyperparameters["activation"], name="fifth_layer_out")(
#             concatenation_layer)

#     return output


# def sixth_level(input_layer, hyperparameters):
#     index = 5
#     c_03 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=3,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_three_"+str(index))(input_layer)
#     c_05 = Conv1D(
#         filters=hyperparameters["filters"][index],
#         kernel_size=5,
#         strides=1,
#         padding=hyperparameters["padding"],
#         kernel_regularizer=regularizers.l2(
#             hyperparameters["kernel_regularizer"]),
#         name="conv_five_"+str(index))(input_layer)

#     concatenation_layer = concatenate([c_03, c_05], axis=-1)
#     max_pool = MaxPooling1D(pool_size=2)(concatenation_layer)

#     if hyperparameters["batch_normalization"]:
#         bn = BatchNormalization(name="bn_"+str(index))(max_pool)
#         output = Activation(
#             hyperparameters["activation"], name="sixth_layer_out")(bn)
#     else:
#         output = Activation(hyperparameters["activation"], name="sixth_layer_out")(
#             concatenation_layer)

#     return output


class DWNet:
    def __init__(self):
        self.data = []

    def get_features(cnn, test_set):
        return cnn.predict(test_set, batch_size=1)

    # create model from hypes dict
    def get_model(input_layer, hyperparameters, visual=False):
        first_out = first_level(input_layer, hyperparameters)
        second_out = second_level(first_out, hyperparameters)
        third_out = third_level(second_out, hyperparameters)
#        fourth_out = fourth_level(third_out,hyperparameters)
#        fifth_out = fifth_level(fourth_out,hyperparameters)
#        sixth_out = sixth_level(fifth_out,hyperparameters)

        # flatten
        flatten = Flatten()(third_out)

        # Neural Network with Dropout
        nn = Dense(units=hyperparameters["neurons"][0],
                   activation=hyperparameters["activation"], name="nn_layer")(flatten)
        do = Dropout(rate=hyperparameters["dropout"], name="drop")(nn)
        if not visual:
            # Classification Layer
            final_layer = Dense(
                units=hyperparameters["num_classes"], activation=hyperparameters["classifier"], name=hyperparameters["classifier"]+"_layer")(do)
        else:
            final_layer = do
        return final_layer
