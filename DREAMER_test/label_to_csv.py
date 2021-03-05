import numpy as np
import scipy.io as sio
import tensorflow as tf
import pandas as pd

print("Label to CSV...")
path = u'DREAMER.mat'
data = sio.loadmat(path)

# 23명의 실험참여자가 18개의 비디오 클립을 시청하고, 128Hz로 60초간 14개의 채널로 측정한 RAW 데이터
numOfSubjects = 23      # 실험 참여자 수
numOfVideo = 18         # 시청한 비디오 클립 수
numOfEEGChannel = 14    # EEG 측정 채널 수
numOfECGChannel = 2     # ECG 측정 채널 수
numOfCategory = 3       # 감성 분류 카테고리 수 (Valence, Arousal, Dominance)

Labels = np.zeros((numOfCategory, numOfSubjects, numOfVideo))

for iter_subject in range(0, numOfSubjects):
    for iter_video in range(0, numOfVideo):

        if data['DREAMER'][0, 0]['Data'][0, iter_subject]['ScoreValence'][0, 0][iter_video, 0] < 4:
            Labels[0, iter_subject, iter_video] = 0
        else:
            Labels[0, iter_subject, iter_video] = 1
        if data['DREAMER'][0, 0]['Data'][0, iter_subject]['ScoreArousal'][0, 0][iter_video, 0] < 4:
            Labels[1, iter_subject, iter_video] = 0
        else:
            Labels[1, iter_subject, iter_video] = 1
        if data['DREAMER'][0, 0]['Data'][0, iter_subject]['ScoreDominance'][0, 0][iter_video, 0] < 4:
            Labels[2, iter_subject, iter_video] = 0
        else:
            Labels[2, iter_subject, iter_video] = 1

Labels = Labels.reshape(numOfCategory, numOfSubjects*numOfVideo)

print(Labels)

Labels_df = pd.DataFrame(Labels)
Labels_df.to_csv("labels.csv", header=False, index=False)
