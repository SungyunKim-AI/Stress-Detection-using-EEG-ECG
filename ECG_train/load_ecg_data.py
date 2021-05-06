import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy import signal
from tqdm import tqdm

class Load_Data:
    
    def __init__(self, home_dir, ch = list(range(3)), fs=512, ss=40, step=2, subejct=9):
        self.Fs = fs    # Sample Frequency (Hz)
        self.Ss = ss     # Sample second (sec)
        self.step = step    # Overlapping Step (sec)
        self.channels = ch
        self.Subjects = subejct
        self.home_dir = home_dir

        # Dataset Path
        self.baseline_paths = "C:/Users/user/Desktop/data_preprocessed/ECG_preprocessed/" + home_dir + "/baseline/"
        self.stimuli_paths = "C:/Users/user/Desktop/data_preprocessed/ECG_preprocessed/" + home_dir + "/stimuli/"

        self.deleteData = [1,2,3,8,9]

    # ============ 각 피실험자당 1개 Test, 1개 Validate 선정해서 오버래핑하고 셔플 ============
    def load_ecg_data(self):
        print("ECG Data Loading...")

        x_Train, x_Test, x_Validate = [], [], []
        y_Train, y_Test, y_Validate = [], [], []

        samples = shuffle(np.array(list(range(1,11))), random_state=42)

        for category, dir_path in enumerate([self.baseline_paths, self.stimuli_paths]):
            for subject in tqdm(range(5, self.Subjects+1)):
                for i, sample in enumerate(samples):

                    if subject == 8 and (sample == 1 or sample == 2 or sample == 3 or sample == 8 or sample == 9):
                        continue

                    path = dir_path + "s" + str(subject) + "_" + str(sample) + ".csv"
                    dataset = pd.read_csv(path, header=None)
                    data_frames = pd.DataFrame(dataset)
                    data = np.array(data_frames.values)
                    # data = np.transpose(data)

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

        print("Train Data Shape : ", x_Train.shape)         # (1175, 3, 5120)
        print("Test Data Shape : ", x_Test.shape)           # (170, 3, 5120)
        print("Validate Data Shape : ", x_Validate.shape)   # (167, 3, 5120)
        print("Train Labels Shape : ", y_Train.shape)       # (1175, 2)
        print("Test Labels Shape : ", y_Test.shape)         # (170, 2)
        print("Validate Labels Shape : ", y_Validate.shape) # (167, 2)

        return x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate



    def STFT(self, train, test, validate, ch=2):
        print("SFTF...")

        x_Train, x_Test, x_Validate = [], [], []

        for i, dataset in enumerate([train, test, validate]):
            for sample in dataset[:, ch]:
                f, t, Zxx = signal.spectrogram(sample, fs=self.Fs)

                if i == 0:
                    x_Train.append(Zxx)
                elif i == 1:
                    x_Test.append(Zxx)
                elif i == 2:
                    x_Validate.append(Zxx)
        
        print("Train Data Shape : ", np.array(x_Train).shape)         # (1175, 3, 5120)
        print("Test Data Shape : ", np.array(x_Test).shape)           # (170, 3, 5120)
        print("Validate Data Shape : ", np.array(x_Validate).shape)   # (167, 3, 5120)

        return x_Train, x_Test, x_Validate






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
data_loader = Load_Data(home_dir="QRS_256", ch=0 ,fs=256)
[x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate] = data_loader.load_ecg_data()
savePath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_QRS_256.npz"
np.savez_compressed(savePath, 
    x_Train=x_Train, x_Test=x_Test, x_Validate=x_Validate, 
    y_Train=y_Train, y_Test=y_Test, y_Validate=y_Validate)

data_loader = Load_Data(home_dir="QRS_128", ch=0, fs=128)
[x_Train, x_Test, x_Validate, y_Train, y_Test, y_Validate] = data_loader.load_ecg_data()
savePath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_QRS_128.npz"
np.savez_compressed(savePath, 
    x_Train=x_Train, x_Test=x_Test, x_Validate=x_Validate, 
    y_Train=y_Train, y_Test=y_Test, y_Validate=y_Validate)

# ecg_dataset_256 = Load_Data(home_dir="FS_256", fs=256)
# [train, test, validate, y_Train, y_Test, y_Validate] =ecg_dataset_256.load_ecg_data()
# [x_Train, x_Test, x_Validate] = ecg_dataset_256.STFT(train, test, validate)
# savePath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_STFT_256.npz"
# np.savez_compressed(savePath, 
#     x_Train=x_Train, x_Test=x_Test, x_Validate=x_Validate, 
#     y_Train=y_Train, y_Test=y_Test, y_Validate=y_Validate)

# Train Data Shape :  (1175, 129, 45)
# Test Data Shape :  (170, 129, 45)
# Validate Data Shape :  (167, 129, 45)
# Train Labels Shape :  (1175, 2)
# Test Labels Shape :  (170, 2)
# Validate Labels Shape :  (167, 2)


# ecg_dataset_128 = Load_Data(home_dir="FS_128", fs=128)
# [train, test, validate, y_Train, y_Test, y_Validate] =ecg_dataset_128.load_ecg_data()
# [x_Train, x_Test, x_Validate] = ecg_dataset_128.STFT(train, test, validate)
# savePath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_STFT_128.npz"
# np.savez_compressed(savePath, 
#     x_Train=x_Train, x_Test=x_Test, x_Validate=x_Validate, 
#     y_Train=y_Train, y_Test=y_Test, y_Validate=y_Validate)

# Train Data Shape :  (1175, 129, 22)
# Test Data Shape :  (170, 129, 22)
# Validate Data Shape :  (167, 129, 22)
# Train Labels Shape :  (1175, 2)
# Test Labels Shape :  (170, 2)
# Validate Labels Shape :  (167, 2)
