import numpy as np
import scipy.io as sio
import tensorflow as tf
import pandas as pd
from glob import glob

# test_data 폴더 안에 데이터셋을 넣고 돌리면 됨.


def test_dataloader():
    print("Test Data Loading...")

    # 23명의 실험참여자가 18개의 비디오 클립을 시청하고, 128Hz로 60초간 14개의 채널로 측정한 RAW 데이터
    numOfSubjects = 23      # 실험 참여자 수
    numOfVideo = 18         # 시청한 비디오 클립 수
    numOfEEGChannel = 14    # EEG 측정 채널 수
    numOfECGChannel = 2     # ECG 측정 채널 수
    numOfCategory = 3       # 감성 분류 카테고리 수 (Valence, Arousal, Dominance)

    print("Label Loading...")
    Labels = pd.read_csv("labels.csv", header=None)
    Labels = np.asarray(Labels)

    print("EEG Data Loading...")
    EEG_stimuli = []
    path_list = glob('./test_data/*')
    print(path_list)
    for path in path_list:
        data = pd.read_csv(path, header=None)  # 각 데이터 셋
        EEG_stimuli.append(data)
    EEG_stimuli = np.asarray(EEG_stimuli)

    print("Labels.shape = ", Labels.shape)  # (3, 414)
    # (dataNum, Channel, 7808)
    print("EEG_stimuli.shape = ", EEG_stimuli.shape)

    return EEG_stimuli, Labels


test_dataloader()
