"""
1. The data was downsampled to 128Hz.
2. EOG artefacts were removed as in [1].
3. A bandpass frequency filter from 4.0-45.0Hz was applied.
4. The data was averaged to the common reference.
5. The EEG channels were reordered so that they all follow the Geneva order as above.
6. The data was segmented into 60 second trials and a 3 second pre-trial baseline removed.
7. The trials were reordered from presentation order to video (Experiment_id) order.
"""

import numpy as np
import pyeeg as pe
import pickle as pickle
import pandas as pd
import math

channel = [1,2,3,4,6,11,13,17,19,20,21,25,29,31] #14 Channels chosen to fit Emotiv Epoch+
band = [4,8,12,16,25,45] #5 bands
window_size = 256 #Averaging band power of 2 sec
step_size = 16 #Each 0.125 sec update once
videos = 40
sample_rate = 128 #Sampling rate of 128 Hz
subjectList = ['01','02','03','04','05','06','07','08','09',
                '10','11','12','13','14','15','16','17','18','19',
                '20','21','22','23','24','25','26','27','28','29',
                '30','31','32']
path_to_dataset = '/Users/user/Desktop/DEAP dataset/data_preprocessed_python/'
save_path = '/Users/user/Desktop/DEAP_dataset2/'


def Preprocessing (channel, band, window_size, step_size, sample_rate):
    print("Preprocessing...")
    data = np.zeros((40, 32, 14, 8064))    # data np array (video/trial x subjects x channel x data)
    labels = np.zeros((40, 32, 4))         # labels np array (label (valence, arousal, dominance, liking) x subjects)
    
    for subIdx in range (0, 32):
        print("Subject ( ", (subIdx + 1), " / 32 )")

        with open(path_to_dataset + 's' + subjectList[subIdx] + '.dat', 'rb') as file:
            subject = pickle.load(file, encoding='latin1')
        
            # subject["data"]   = 40 x 40 x 8064 (video/trial x channel x data)
            # subject["labels"] = 40 x 4         (video/trial x label (valence, arousal, dominance, liking))      
            for videoIdx in range (0, videos):
                tempData = np.array(subject["data"][videoIdx])
                tempLabels = np.array(subject["labels"][videoIdx])
                tempLabels[tempLabels < 5] = 0
                tempLabels[tempLabels >= 5] = 1
                labels[videoIdx][subIdx][:] = tempLabels

                for i, channelIdx in enumerate(channel):
                    data[videoIdx][subIdx][i][:] = tempData[channelIdx][:]
    
    # save data        
    for i in range (0, 40):
        fileName = save_path + 'labels/labels' + str(i+1) + '.csv'
        np.savetxt(fileName, labels[i], delimiter=",")
        # df_labels = pd.DataFrame(labels[i])
        # df_labels.to_csv(fileName, index=False)

        for j in range (0, 32):
            fileName = save_path + 'data/data' + str(i+1) + '_' + str(j+1) + '.csv'
            np.savetxt(fileName, data[i][j], delimiter=",")
            # df = pd.DataFrame(data[i][j])
            # df.to_csv(fileName, index=False)
    

Preprocessing(channel, band, window_size, step_size, sample_rate)
