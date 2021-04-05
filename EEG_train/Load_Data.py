import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import glob
from tqdm import tqdm

# 전체 세트 오버래핑

class Load_Data:

    def __init__(self):
        self.Fs = 128    # Sample Frequency (Hz)
        self.Ss = 40     # Sample second (sec)
        self.step = 2    # Overlapping Step (sec)
        self.channels = list(range(13))
        self.Subjects = 10

        # Dataset Path _ ASR + CAR
        self.baseline_paths = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/baseline/alpha/"
        self.stimuli_paths = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/stimuli/alpha/"

        # Dataset Path _ CAR
        # self.baseline_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/CAR_preprocessed/EEG/baseline/alpha/*")
        # self.stimuli_paths = glob.glob("C:/Users/user/Desktop/data_preprocessed/CAR_preprocessed/EEG/stimuli/alpha/*")

    # ======================= 전체 데이터 셋을 오버래핑 ========================
    def load_eeg_data(self):
        print("EEG Data Loading...")

        EEG = []
        Labels = []

        numOfBaseline = 0
        numOfStimuli = 0
        for category in [baseline_paths, stimuli_paths]:
            for path in tqdm(category):
                dataset = pd.read_csv(path, header=None)
                data_frames = pd.DataFrame(dataset)
                data = np.array(data_frames.values)

                # data overlapping
                for i in range(0, int(data.shape[1]/Fs) - Ss + 1, step):
                    part_data = data[0:13, (i*Fs) : ((i+Ss)*Fs)]
                    EEG.append(part_data)

                    # Labels one-hot encoding
                    if category == baseline_paths:
                        Labels.append([1, 0])     
                        numOfBaseline += 1  
                    elif category == stimuli_paths:
                        Labels.append([0, 1])
                        numOfStimuli += 1
            
        EEG = np.array(EEG)
        Labels = np.array(Labels)

        print("EEG Data Shape : ", EEG.shape)       # (3032, )
        print("Labels Shape : ", Labels.shape)      # (3032, 2)
        print("numOfBaseline : ", numOfBaseline)    # 1831
        print("numOfStimuli : ", numOfStimuli)      # 1201
        print("Samples : ", samples)

        return EEG, Labels, numOfBaseline, numOfStimuli, samples


    # ======== Train, Test, Validate 미리 나눠서 개별적으로 오버레핑하고 셔플 ==========
    def load_eeg_data_2(self):
        print("EEG Data Loading...")

        x_Train = []
        x_Test = []
        x_Validate = []
        y_Train = []
        y_Test =[]
        y_Validate = []

        for category, dir_path in enumerate([self.baseline_paths, self.stimuli_paths]):
            for subject in tqdm(range(1, self.Subjects+1)):
                if subject == 2:
                    continue

                for sample in range(1, 11):
                    path = dir_path + "s" + str(subject) + "_" + str(sample) + ".csv"

                    dataset = pd.read_csv(path, header=None)
                    data_frames = pd.DataFrame(dataset)
                    data = np.array(data_frames.values)

                    if subject == 10:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, self.channels, category, subject)
                        x_Validate.extend(overlappedData)
                        y_Validate.extend(overlappedLabel)
                    elif subject == 9:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, self.channels, category, subject)
                        x_Test.extend(overlappedData)
                        y_Test.extend(overlappedLabel)
                    else:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, self.channels, category, subject)
                        x_Train.extend(overlappedData)
                        y_Train.extend(overlappedLabel)

        # Data Shuffle
        [x_Train, y_Train] = shuffle(np.array(x_Train), np.array(y_Train))
        [x_Test, y_Test] = shuffle(np.array(x_Test), np.array(y_Test))
        [x_Validate, y_Validate] = shuffle(np.array(x_Validate), np.array(y_Validate))

        print("Train Data Shape : ", x_Train.shape)         # (2868, 13, 5120)
        print("Test Data Shape : ", x_Test.shape)           # (394, 13, 5120)
        print("Validate Data Shape : ", x_Validate.shape)   # (400, 13, 5120)
        print("Train Labels Shape : ", y_Train.shape)       # (2868, 2)
        print("Test Labels Shape : ", y_Test.shape)         # (394, 2)
        print("Validate Labels Shape : ", y_Validate.shape) # (400, 2)

        return x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate


    def data_overlapping(self, data, chans, category, subject):
        overlappedData = []
        overlappedLabel = []
        endPoint = int(data.shape[1]/self.Fs) - self.Ss + 1
        for i in range(0, endPoint, self.step):
            start = i * self.Fs
            end = (i + self.Ss) * self.Fs
            part_data = data[chans, start:end]

            overlappedData.append(part_data)
            overlappedLabel.append(self.label_append(category, subject))

        return overlappedData, overlappedLabel

    def label_append(self, category, subject):
        # Labels one-hot encoding
        if category == 0:       # 0: baseline
            return [1, 0]
        elif category == 1:     # 1: stimuli
            return [0, 1]

# [EEG, Labels, numOfBaseline, numOfStimuli, samples] = load_eeg_data()
# [x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate] = Load_Data().load_eeg_data_2()

"""
Train Data Shape :  (335, 13, 5120)
Test Data Shape :  (318, 13, 5120)
Validate Data Shape :  (330, 13, 5120)
Train Labels Shape :  (335, 2)
Test Labels Shape :  (318, 2)
Validate Labels Shape :  (330, 2)
"""
# import matplotlib.pyplot as plt
# fig = plt.figure()
# train = fig.add_subplot(3, 1, 1)
# test = fig.add_subplot(3, 1, 2)
# validate = fig.add_subplot(3, 1, 3)

# train.plot(np.transpose(x_Train[0]))
# test.plot(np.transpose(x_Test[0]))
# validate.plot(np.transpose(x_Validate[0]))
# print(y_Train)
# print(y_Test)
# print(y_Validate)

# plt.show()

