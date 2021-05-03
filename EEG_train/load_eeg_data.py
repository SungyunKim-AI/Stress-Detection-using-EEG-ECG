import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

# 전체 세트 오버래핑

class Load_Data:

    def __init__(self):
        self.Fs = 128    # Sample Frequency (Hz)
        self.Ss = 40     # Sample second (sec)
        self.step = 2    # Overlapping Step (sec)
        self.channels = list(range(14))
        self.Subjects = 9

        # Dataset Path _ ASR + CAR
        # self.baseline_paths = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/baseline/overall/"
        # self.stimuli_paths = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/stimuli/overall/"
        # self.baseline_paths = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/baseline/alpha/"
        # self.stimuli_paths = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/stimuli/alpha/"

        # Dataset Path _ CAR
        self.baseline_paths = "C:/Users/user/Desktop/data_preprocessed/CAR_preprocessed/EEG/baseline/overall/"
        self.stimuli_paths = "C:/Users/user/Desktop/data_preprocessed/CAR_preprocessed/EEG/stimuli/overall/"

    # ============ 각 피실험자당 1개 Test, 1개 Validate 선정해서 오버래핑하고 셔플 ============
    def load_eeg_data(self):
        print("EEG Data Loading...")

        x_Train = []
        x_Test = []
        x_Validate = []
        y_Train = []
        y_Test = []
        y_Validate = []

        samples = shuffle(np.array(list(range(1,11))), random_state=42)

        for category, dir_path in enumerate([self.baseline_paths, self.stimuli_paths]):
            for subject in tqdm(range(1, self.Subjects+1)):
                for i, sample in enumerate(samples):
                    path = dir_path + "s" + str(subject) + "_" + str(sample) + ".csv"
                    dataset = pd.read_csv(path, header=None)
                    data_frames = pd.DataFrame(dataset)
                    data = np.array(data_frames.values)

                    if i == 8:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, self.channels, category, subject)
                        x_Validate.extend(overlappedData)
                        y_Validate.extend(overlappedLabel)
                    elif i == 9:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, self.channels, category, subject)
                        x_Test.extend(overlappedData)
                        y_Test.extend(overlappedLabel)
                    else:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, self.channels, category, subject)
                        x_Train.extend(overlappedData)
                        y_Train.extend(overlappedLabel)

        # Data Shuffle
        [x_Train, y_Train] = shuffle(np.array(x_Train), np.array(y_Train), random_state=42)
        [x_Test, y_Test] = shuffle(np.array(x_Test), np.array(y_Test), random_state=42)
        [x_Validate, y_Validate] = shuffle(np.array(x_Validate), np.array(y_Validate), random_state=42)

        print("Train Data Shape : ", x_Train.shape)         # (2543, 14, 5120)
        print("Test Data Shape : ", x_Test.shape)           # (319, 14, 5120)
        print("Validate Data Shape : ", x_Validate.shape)   # (321, 14, 5120)
        print("Train Labels Shape : ", y_Train.shape)       # (2543, 2)
        print("Test Labels Shape : ", y_Test.shape)         # (319, 2)
        print("Validate Labels Shape : ", y_Validate.shape) # (321, 2)

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



# ================= Save Dataset Numpy format =====================
[x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate] = Load_Data().load_eeg_data()

savePath = "C:/Users/user/Desktop/numpy_dataset/eeg_dataset.npz"
np.savez_compressed(savePath, 
    x_Train=x_Train, x_Test=x_Test, x_Validate=x_Validate, 
    y_Train=y_Train, y_Test=y_Test, y_Validate=y_Validate)
