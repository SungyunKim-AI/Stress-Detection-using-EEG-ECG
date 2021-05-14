import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

# 전체 세트 오버래핑

class Load_Data:

    def __init__(self, home_dir, sampleLength, step):
        self.Fs = 128    # Sample Frequency (Hz)
        self.Ss = sampleLength     # Sample second (sec)
        self.step = step    # Overlapping Step (sec)
        self.channels = list(range(13))
        self.Subjects = 19
        self.home_dir = home_dir

        # Dataset Path_EEG
        self.baseline_paths_EEG = "C:/Users/user/Desktop/data_preprocessed/" + home_dir + "/EEG/baseline/alpha/"
        self.stimuli_paths_EEG = "C:/Users/user/Desktop/data_preprocessed/" + home_dir + "/EEG/stimuli/alpha/"
        
        # Dataset Path_ECG
        self.baseline_paths_ECG = "C:/Users/user/Desktop/data_preprocessed/band_filter_preprocessed/" + home_dir + "/baseline/"
        self.stimuli_paths_ECG = "C:/Users/user/Desktop/data_preprocessed/band_filter_preprocessed/" + home_dir + "/stimuli/"


    # Load EEG Data
    def load_eeg_data(self):
        print("EEG Data Loading...")
        x_Train, x_Test = [], [], []
        y_Train, y_Test = [], []

        for category, dir_path in enumerate([self.baseline_paths_EEG, self.stimuli_paths_EEG]):
            for subject in tqdm(range(5, self.Subjects+1)):
                for sample in range(1,11):

                    if subject == 12 and (sample in [6,7,8,9,10]):
                        continue

                    path = dir_path + "s" + str(subject) + "_" + str(sample) + ".csv"
                    dataset = pd.read_csv(path, header=None)
                    data_frames = pd.DataFrame(dataset)
                    data = np.array(data_frames.values)

                    if sample == 4 or sample == 8:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, category, subject)
                        x_Test.extend(overlappedData)
                        y_Test.extend(overlappedLabel)
                    else:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, category, subject)
                        x_Train.extend(overlappedData)
                        y_Train.extend(overlappedLabel)

        # Data Shuffle
        [x_Train, y_Train] = shuffle(np.array(x_Train), np.array(y_Train), random_state=42)
        [x_Test, y_Test] = shuffle(np.array(x_Test), np.array(y_Test), random_state=42)
        [x_Validate, y_Validate] = shuffle(np.array(x_Validate), np.array(y_Validate), random_state=42)

        print("Train Data Shape : ", x_Train.shape)         # (2543, 14, 5120)
        print("Test Data Shape : ", x_Test.shape)           # (319, 14, 5120)
        print("Train Labels Shape : ", y_Train.shape)       # (2543, 2)
        print("Test Labels Shape : ", y_Test.shape)         # (319, 2)

        return x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate

    # Load ECG Data
    def load_ecg_data(self):
        print("ECG Data Loading...")
        x_Train, x_Test, x_Validate = [], [], []
        y_Train, y_Test, y_Validate = [], [], []

        for category, dir_path in enumerate([self.baseline_paths_ECG, self.stimuli_paths_ECG]):
            for subject in tqdm(range(5, self.Subjects+1)):
                for sample in range(1,11):

                    path = dir_path + "s" + str(subject) + "_" + str(sample) + ".csv"
                    dataset = pd.read_csv(path, header=None)
                    data_frames = pd.DataFrame(dataset)
                    data = np.array(data_frames.values)
                    data = np.transpose(data)

                    if sample == 4:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, category, subject)
                        x_Validate.extend(overlappedData)
                        y_Validate.extend(overlappedLabel)
                    elif sample == 8:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, category, subject)
                        x_Test.extend(overlappedData)
                        y_Test.extend(overlappedLabel)
                    else:
                        [overlappedData, overlappedLabel] = self.data_overlapping(data, category, subject)
                        x_Train.extend(overlappedData)
                        y_Train.extend(overlappedLabel)

        # Data Shuffle
        [x_Train, y_Train] = shuffle(np.array(x_Train), np.array(y_Train), random_state=42)
        [x_Test, y_Test] = shuffle(np.array(x_Test), np.array(y_Test), random_state=42)
        [x_Validate, y_Validate] = shuffle(np.array(x_Validate), np.array(y_Validate), random_state=42)

        print("Train Data Shape : ", x_Train.shape)      
        print("Test Data Shape : ", x_Test.shape)          
        print("Validate Data Shape : ", x_Validate.shape)   
        print("Train Labels Shape : ", y_Train.shape)       
        print("Test Labels Shape : ", y_Test.shape)        
        print("Validate Labels Shape : ", y_Validate.shape) 

        return x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate

    # Cut and Overlap Data
    def data_overlapping(self, data, category, subject):
        overlappedData = []
        overlappedLabel = []
        endPoint = (data.shape[1]//self.Fs) - self.Ss + 1
        for i in range(0, endPoint, self.step):
            start = i * self.Fs
            end = (i + self.Ss) * self.Fs
            part_data = data[:, start:end]

            # Data Normalize
            part_data = (part_data - part_data.mean()) / (part_data.std())

            overlappedData.append(part_data)
            overlappedLabel.append(self.label_append(category, subject))

        return overlappedData, overlappedLabel

    # Labels one-hot encoding
    def label_append(self, category, subject):
        if category == 0:       # 0: baseline
            return [1, 0]
        elif category == 1:     # 1: stimuli
            return [0, 1]



# ================= Save Dataset Numpy format =====================
data_loader = Load_Data(home_dir="ASR_CAR_preprocessed", sampleLength=30, step=2)
[x_Train, x_Test, y_Train, y_Test] = data_loader.load_eeg_data()
savePath = "C:/Users/user/Desktop/numpy_dataset/eeg_dataset_ASR_alpha.npz"
# np.savez_compressed(savePath, 
#     x_Train=x_Train, x_Test=x_Test, x_Validate=x_Validate, 
#     y_Train=y_Train, y_Test=y_Test, y_Validate=y_Validate)


data_loader = Load_Data(home_dir="ECG_128", fs=128)
[x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate] = data_loader.load_ecg_data()
savePath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_128_norm.npz"
# np.savez_compressed(savePath, 
#     x_Train=x_Train, x_Test=x_Test, x_Validate=x_Validate, 
#     y_Train=y_Train, y_Test=y_Test, y_Validate=y_Validate)
