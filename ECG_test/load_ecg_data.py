import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt


# accuracy 그래프 그리기
def plot(data):
    plt.figure()
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.plot(data)
    plt.show()


def load_ecg_data():
    print("ECG Data Loading...")

    numOfData = 40

    Labels = []
    ECG = []

    # 데이터 경로 설정
    baseline_paths = glob.glob(
        "C:/Users/kia34/문서/GitHub/Graduation-Project/ECG_test/ECG/baseline/*")
    stimuli_paths = glob.glob(
        "C:/Users/kia34/문서/GitHub/Graduation-Project/ECG_test/ECG/stimuli/*")

    for path in baseline_paths:
        data = pd.read_csv(path, header=None)
        data = data[:2900][0]
        ECG.append(data)
        Labels.append([1, 0])

    for path in stimuli_paths:
        data = pd.read_csv(path, header=None)
        data = data[:2900][0]
        ECG.append(data)
        Labels.append([0, 1])

    ECG = np.array(ECG)
    Labels = np.array(Labels)

    print(ECG.shape)
    print(Labels.shape)

    return ECG, Labels
    # for videoIdx in tqdm(range(0, numOfVideo), desc='VideoIdx'):
    #     fileName_labels = path_labelset + str(videoIdx+1) + ".csv"
    #     labels = pd.read_csv(fileName_labels, header=None)      # 32 x 4
    #     labels = np.array(labels)

    #     for subIdx in range(0, numOfSubjects):
    #         fileName_data = path_dataset + \
    #             str(videoIdx+1) + "_" + str(subIdx+1) + ".csv"
    #         eeg = pd.read_csv(fileName_data, header=None)       # 14 x 8064
    #         EEG.append(np.array(eeg))
    #         Labels.append(labels[subIdx])

    # Labels = np.array(Labels).T
    # EEG = np.array(EEG)
    # print("Labels.shape = ", Labels.shape)    # (1280, 4)
    # print("EEG.shape = ", EEG.shape)          # (1280, 14, 8064)

    # return EEG, Labels


# test_dataloader()
load_ecg_data()
