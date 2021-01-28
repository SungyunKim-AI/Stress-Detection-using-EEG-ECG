import numpy as np
import pandas as pd
from scipy import io
from sklearn import preprocessing as pre

if __name__ == '__main__':
    total=0
    path=u'DREAMER.mat'
    data=io.loadmat(path)
    print("EEG signals are being feature extracted...")
    EEG_tmp=np.zeros((23,18,42))
    for k in range(0,23):
        for j in range(0,18):
            for i in range(0,14):
                B,S=[],[]
                basl=data['DREAMER'][0,0]['Data'][0,k]['EEG'][0,0]['baseline'][0,0][j,0][:,i]
                stim=data['DREAMER'][0,0]['Data'][0,k]['EEG'][0,0]['stimuli'][0,0][j,0][:,i]
                B=preprocessing(basl,B)
                S=preprocessing(stim,S)`
                Extrod=np.divide(S,B)
                total+=1
                EEG_tmp[k,j,3*i]=Extrod[0]
                EEG_tmp[k,j,3*i+1]=Extrod[1]
                EEG_tmp[k,j,3*i+2]=Extrod[2]
                print("\rprogress: %d%%" %(total/(23*18*14)*100),end="")
    col=[]
    for i in range(0,14):
        col.append('psdtheta_'+str(i + 1)+'_un')
        col.append('psdalpha_'+str(i + 1)+'_un')
        col.append('psdbeta_'+str(i + 1)+'_un')
    EEG=pd.DataFrame(EEG_tmp.reshape((23 * 18,EEG_tmp.shape[2])),columns=col)
    scaler=pre.StandardScaler()
    for i in range(len(col)):
        EEG[col[i][:-3]]=scaler.fit_transform(EEG[[col[i]]])
    EEG.drop(col,axis=1,inplace=True)
    print(EEG)
    EEG.to_csv('Extracted_EEG.csv')
