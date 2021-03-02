import numpy as np
import pandas as pd
from tqdm import tqdm

# test_data 폴더 안에 데이터셋을 넣고 돌리면 됨.
path_dataset = '/Users/user/Desktop/DEAP_dataset2/data/data'
path_labelset = '/Users/user/Desktop/DEAP_dataset2/labels/labels'

def test_dataloader():
    print("Data Loading...")

    # # 23명의 실험참여자가 18개의 비디오 클립을 시청하고, 128Hz로 60초간 14개의 채널로 측정한 RAW 데이터
    # numOfSubjects = 23      # 실험 참여자 수
    # numOfVideo = 18         # 시청한 비디오 클립 수
    # numOfEEGChannel = 14    # EEG 측정 채널 수
    # numOfECGChannel = 2     # ECG 측정 채널 수
    # numOfCategory = 3       # 감성 분류 카테고리 수 (Valence, Arousal, Dominance)

    # 32명의 실험참여자가 40개의 비디오 클립을 시청하고, 128Hz로 60초간 14개의 채널로 측정한 RAW 데이터
    numOfSubjects = 32      # 실험 참여자 수
    numOfVideo = 40         # 시청한 비디오 클립 수
    numOfEEGChannel = 14    # EEG 측정 채널 수
    numOfECGChannel = 2     # ECG 측정 채널 수
    numOfCategory = 3       # 감성 분류 카테고리 수 (Valence, Arousal, Dominance)


    Labels = []     # 4 x (32*40)
    EEG = []        # (32*40) x 14 x 8064

    for videoIdx in tqdm(range (0, numOfVideo), desc='VideoIdx'):
        fileName_labels = path_labelset + str(videoIdx+1) + ".csv"
        labels = pd.read_csv(fileName_labels, header=None)      # 32 x 4
        labels = np.array(labels)

        for subIdx in range (0, numOfSubjects):
            fileName_data = path_dataset + str(videoIdx+1) + "_" + str(subIdx+1) + ".csv"
            eeg = pd.read_csv(fileName_data, header=None)       # 14 x 8064
            EEG.append(np.array(eeg))
            Labels.append(labels[subIdx])

    Labels = np.array(Labels).T
    EEG = np.array(EEG)
    print("Labels.shape = ", Labels.shape)    # (1280, 4)
    print("EEG.shape = ", EEG.shape)          # (1280, 14, 8064)

    return EEG, Labels


#test_dataloader()
