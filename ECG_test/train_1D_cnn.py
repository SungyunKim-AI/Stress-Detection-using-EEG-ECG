#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eleftherios
@github: https://github.com/trivizakis
"""
from data_augmentation_timeseries import DataAugmentation
from model import CustomModel as Model
import numpy as np
from utils import Utils
from sklearn.model_selection import StratifiedShuffleKFold, StratifiedShuffleShuffleSplit
import keras
from keras import backend as K
import scipy.io
import math
from sklearn.utils import shuffle


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


def getPaddedSlice(npArray, pos, lenSegment, center=False):
    lenNpArray = len(npArray)
    if center:
        if lenSegment % 2 == 0:
            startIndex = int(pos - math.floor(lenSegment / 2.0)) + 1
            lastIndex = int(pos + math.ceil(lenSegment / 2.0)) + 1

        else:
            startIndex = int(pos - math.floor(lenSegment / 2.0))
            lastIndex = int(pos + math.ceil(lenSegment / 2.0)) + 1
    else:
        startIndex = pos
        lastIndex = startIndex + lenSegment

    if startIndex < 0:
        padded_slice = npArray[0: lastIndex]
        padded_slice = np.concatenate(
            (np.zeros(abs(startIndex)), padded_slice))
    else:
        if center:
            padded_slice = npArray[startIndex: lastIndex]
        else:
            padded_slice = npArray[pos: lastIndex]

    if lastIndex > lenNpArray:
        if center:
            padded_slice = npArray[startIndex: pos + lenSegment]
            padded_slice = np.concatenate(
                (padded_slice, np.zeros(lastIndex - lenNpArray)))
        else:
            padded_slice = npArray[pos: pos + lenSegment]
            padded_slice = np.concatenate(
                (padded_slice, np.zeros(lastIndex - lenNpArray)))

    return padded_slice


def slicer(list_to_slice, studied_window):
    slice_step = 50
    temp = []

    for index in range(len(list_to_slice)):
        for pos in range(0, len(list_to_slice[index]), slice_step):
            temp.append(getPaddedSlice(np.hstack(
                list_to_slice[index]), pos, studied_window, center=False).reshape(-1, 1))
    return temp

# experiment specific data extraction (matlab files)


def data_extractor(data, studied_window):
    stress = []
    neutral = []
    for subject in data:
        for index in range(0, len(subject)):
            if index == 0 or index == 2 or index == 8 or index == 9:
                #            neutral or relaxed(9)
                neutral.append(subject[index])
            elif index == 6:
                #            fatigue
                continue
            else:
                #            stress
                stress.append(subject[index])

    stress_final = slicer(stress, studied_window)
    neutral_final = slicer(neutral, studied_window)

    stress_labels = np.ones(len(stress_final), int)
    neutral_labels = np.zeros(len(neutral_final), int)

    dataset = np.array(stress_final + neutral_final)
    labels = np.concatenate((stress_labels, neutral_labels))

    return shuffle(dataset, labels)


hyperparameters = Utils.get_hypes()

data = scipy.io.loadmat(hyperparameters["dataset_dir"]+"stress.mat")

# crossvalidation
tst_split_index = 1
val_split_index = 1
skf_tr_tst = StratifiedKFold(
    n_splits=hyperparameters["kfold"][0], shuffle=hyperparameters["shuffle"])
skf_tr_val = StratifiedShuffleSplit(
    n_splits=hyperparameters["kfold"][1], test_size=0.17, train_size=0.83)
for trval_index, tst_index in skf_tr_tst.split(data["HRV1"]):
    convergence_data = data["HRV1"][trval_index]
    for tr_index, val_index in skf_tr_val.split(convergence_data):

        data_tr, labels_tr = data_extractor(
            convergence_data[tr_index], hyperparameters["input_shape"][0])
        data_val, labels_val = data_extractor(
            convergence_data[val_index], hyperparameters["input_shape"][0])
        data_tst, labels_tst = data_extractor(
            data["HRV1"][tst_index], hyperparameters["input_shape"][0])

        # offline data augmentation
        if hyperparameters["data_augmentation"] == True:
            # apply data augmentation
            data_tr, labels_tr = DataAugmentation.apply(
                data_tr, labels_tr, hyperparameters)

        # normalize
        dataset = np.concatenate((data_tr, data_val, data_tst))

        if hyperparameters["image_normalization"] == "01":
            x_min = np.amin(dataset.ravel())
            x_max = np.amax(dataset.ravel())
            data_tr = (data_tr - x_min)/(x_max-x_min)
            data_val = (data_val - x_min)/(x_max-x_min)
            data_tst = (data_tst - x_min)/(x_max-x_min)
        elif hyperparameters["image_normalization"] == "-11":
            std = np.std(dataset.ravel())
            mean = np.mean(dataset.ravel())
            data_tr = (data_tr - mean)/std
            data_val = (data_val - mean)/std
            data_tst = (data_tst - mean)/std

        # clear session in every iteration
        K.clear_session()

        # network version
        version = str(tst_split_index)+"."+str(val_split_index)
        hyperparameters["version"] = "network_version:"+version+"/"

        # make dirs
        Utils.make_dirs(version, hyperparameters)

        if hyperparameters["loss"] != "scc":
            labels_tr = keras.utils.to_categorical(
                labels_tr, num_classes=hyperparameters["num_classes"])
            labels_val = keras.utils.to_categorical(
                labels_val, num_classes=hyperparameters["num_classes"])
            labels_tst = keras.utils.to_categorical(
                labels_tst, num_classes=hyperparameters["num_classes"])

        # save patient id per network version
        Utils.save_skf_pids(version+".dataset", data_tr,
                            data_val, data_tst, hyperparameters)
        Utils.save_skf_pids(version+".labels", labels_tr,
                            labels_val, labels_tst, hyperparameters)
        print("data_tr len : ", len(data_tr))
        # create network
        cnn = Model.get_model(hyperparameters)

        # fit network
        cnn = Model.fit_model(cnn, hyperparameters, data_tr,
                              labels_tr, data_val, labels_val)

        # test set performance
        Model.test_fitted_model(cnn, hyperparameters, data_tst, labels_tst)

        # save current hypes
        Utils.save_hypes(
            hyperparameters["chkp_dir"]+hyperparameters["version"], "hypes"+version, hyperparameters)
        # Update version indeces
        val_split_index += 1
    val_split_index = 1
    tst_split_index += 1
