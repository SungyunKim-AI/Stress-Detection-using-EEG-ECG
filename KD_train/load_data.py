import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

# 전체 세트 오버래핑

class Load_Data:

    def __init__(self, sampleLength, step):
        self.Fs = 128              # Sample Frequency (Hz)
        self.Ss = sampleLength     # Sample second (sec)
        self.step = step           # Overlapping Step (sec)
        self.Subjects = 19

        # Dataset Path
        self.paths_EEG = "C:/Users/user/Desktop/data_preprocessed/ASR_CAR_preprocessed/EEG/alpha/"
        self.paths_ECG = "C:/Users/user/Desktop/data_preprocessed/band_filter_preprocessed/ECG_128/"

    def load_data(self):
        print("Data Loading...")
        x_Train_EEG, x_Test_EEG, x_Validate_EEG = [], [], []
        y_Train_EEG, y_Test_EEG, y_Validate_EEG = [], [], []
        x_Train_ECG, x_Test_ECG, x_Validate_ECG = [], [], []
        y_Train_ECG, y_Test_ECG, y_Validate_ECG = [], [], []

        for category in ["baseline", "stimuli"]:
            for subject in tqdm(range(5, self.Subjects+1)):
                for sample in range(1,11):

                    # Missing Data
                    if subject == 12 and (sample in [6,7,8,9,10]):
                        continue
                    
                    # Load EEG data
                    path = self.paths_EEG + category + "/s" + str(subject) + "_" + str(sample) + ".csv"
                    dataset = pd.read_csv(path, header=None)
                    data_EEG = pd.DataFrame(dataset).values

                    # Load ECG Data
                    path = self.paths_ECG + category + "/s" + str(subject) + "_" + str(sample) + ".csv"
                    dataset = pd.read_csv(path, header=None)
                    data_ECG = pd.DataFrame(dataset).values
                    data_ECG = np.transpose(data_ECG)

                    [overlappedData_EEG, overlappedLabel_EEG, overlappedData_ECG, overlappedLabel_ECG] = self.data_overlapping(data_EEG, data_ECG, category, subject)
                    if sample == 8:
                        x_Validate_EEG.extend(overlappedData_EEG)
                        y_Validate_EEG.extend(overlappedLabel_EEG)
                        x_Validate_ECG.extend(overlappedData_ECG)
                        y_Validate_ECG.extend(overlappedLabel_ECG)
                    elif sample == 4:
                        x_Test_EEG.extend(overlappedData_EEG)
                        y_Test_EEG.extend(overlappedLabel_EEG)
                        x_Test_ECG.extend(overlappedData_ECG)
                        y_Test_ECG.extend(overlappedLabel_ECG)
                    else:
                        x_Train_EEG.extend(overlappedData_EEG)
                        y_Train_EEG.extend(overlappedLabel_EEG)
                        x_Train_ECG.extend(overlappedData_ECG)
                        y_Train_ECG.extend(overlappedLabel_ECG)
                    
        # Data Shuffle
        # [x_Train, y_Train] = shuffle(np.array(x_Train), np.array(y_Train), random_state=42)
        # [x_Test, y_Test] = shuffle(np.array(x_Test), np.array(y_Test), random_state=42)
        
        x_Train_EEG = np.array(x_Train_EEG)
        x_Test_EEG = np.array(x_Test_EEG)
        x_Validate_EEG = np.array(x_Validate_EEG)
        y_Train_EEG = np.array(y_Train_EEG)
        y_Test_EEG = np.array(y_Test_EEG)
        y_Validate_EEG = np.array(y_Validate_EEG)

        x_Train_ECG = np.array(x_Train_ECG)
        x_Test_ECG = np.array(x_Test_ECG)
        x_Validate_ECG = np.array(x_Validate_ECG)
        y_Train_ECG = np.array(y_Train_ECG)
        y_Test_ECG = np.array(y_Test_ECG)
        y_Validate_ECG = np.array(y_Validate_ECG)

        print("EEG Data Shape")
        print("Train Data Shape : ", x_Train_EEG.shape)         # 
        print("Test Data Shape : ", x_Test_EEG.shape)           # 
        print("Validate Data Shape : ", y_Validate_EEG.shape)   # 
        print("Train Labels Shape : ", y_Train_EEG.shape)       # 
        print("Test Labels Shape : ", y_Test_EEG.shape)         # 
        print("Validate Labels Shape : ", y_Validate_EEG.shape) # 

        print("ECG Data Shape")
        print("Train Data Shape : ", x_Train_ECG.shape)         # 
        print("Test Data Shape : ", x_Test_ECG.shape)           # 
        print("Validate Data Shape : ", y_Validate_ECG.shape)   # 
        print("Train Labels Shape : ", y_Train_ECG.shape)       # 
        print("Test Labels Shape : ", y_Test_ECG.shape)         # 
        print("Validate Labels Shape : ", y_Validate_ECG.shape) # 

        return x_Train_EEG, x_Test_EEG, x_Validate_EEG, y_Train_EEG, y_Test_EEG, y_Validate_EEG, x_Train_ECG, x_Test_ECG, x_Validate_ECG, y_Train_ECG, y_Test_ECG, y_Validate_ECG

    # Cut and Overlap Data
    def data_overlapping(self, data_EEG, data_ECG, category, subject):
        overlappedData_EEG, overlappedLabel_EEG = [], []
        overlappedData_ECG, overlappedLabel_ECG = [], []

        minLen = min(data_ECG.shape[1], data_EEG.shape[1])
        endPoint = (minLen//self.Fs) - self.Ss + 1
        for i in range(0, endPoint, self.step):
            start = i * self.Fs
            end = (i + self.Ss) * self.Fs
            part_EEG = data_EEG[list(range(13)), start:end]
            part_ECG = data_ECG[2, start:end]

            # Data Normalize
            part_ECG = (part_ECG - part_ECG.mean()) / (part_ECG.std())

            overlappedData_EEG.append(part_EEG)
            overlappedLabel_EEG.append(self.label_append(category))

            overlappedData_ECG.append(part_ECG)
            overlappedLabel_ECG.append(self.label_append(category))

        return overlappedData_EEG, overlappedLabel_EEG, overlappedData_ECG, overlappedLabel_ECG

    # Labels one-hot encoding
    def label_append(self, category):
        if category == "baseline":
            return [1, 0]
        elif category == "stimuli":
            return [0, 1]



# ================= Save Dataset Numpy format =====================
if __name__ == "__main__":
    data_loader = Load_Data(sampleLength=30, step=2)
    [x_Train_EEG, x_Test_EEG, x_Validate_EEG, y_Train_EEG, y_Test_EEG, y_Validate_EEG, x_Train_ECG, x_Test_ECG, x_Validate_ECG, y_Train_ECG, y_Test_ECG, y_Validate_ECG] = data_loader.load_data()

    savePath = "C:/Users/user/Desktop/numpy_dataset/eeg_dataset_ASR_alpha.npz"
    np.savez_compressed(savePath, 
        x_Train=x_Train_EEG, x_Test=x_Test_EEG, x_Validate=x_Validate_EEG, 
        y_Train=y_Train_EEG, y_Test=y_Test_EEG, y_Validate=y_Validate_EEG)

    savePath = "C:/Users/user/Desktop/numpy_dataset/ecg_dataset_128_norm.npz"
    np.savez_compressed(savePath, 
        x_Train=x_Train_ECG, x_Test=x_Test_ECG, x_Validate=x_Validate_ECG, 
        y_Train=y_Train_ECG, y_Test=y_Test_ECG, y_Validate=y_Validate_ECG)

