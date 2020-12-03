import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

frameNum = 60

# load data
dfs = []
for i in range(1,33):
  for j in range(1,41):
    filename = './CWT/File_%dframe_exscale/participant%dvideo%d.txt'%(frameNum,i,j)
    cols = [i for i in range(frameNum)]
    df = pd.read_csv(filename, header = None, usecols = cols, delimiter=',') #data frame  
    dfs.append(df.values)                                                    #data frames
    #print('participant%dvideo%d.txt'%(i,j))

dfs = np.array(dfs)
print('dataLoaded:', dfs.shape)

# normalize
x_min = dfs.min(axis = (1,2),keepdims=True)
x_max = dfs.max(axis = (1,2),keepdims=True)
dfs_normal = (dfs-x_min)/(x_max-x_min)

depth = 3
# divide frames ,or 60s is too long for a single 3d input
reshape_dfs = np.split(dfs_normal, frameNum/depth, axis=2)      #axis=2 -> 3차원 분리
reshape_dfs = np.array(reshape_dfs)
reshape_dfs = np.reshape(reshape_dfs,[-1,1024,depth])           #[-1, 1024, 3] reshape
print("reshape_dfs.shape: ", reshape_dfs.shape)

# load label
cols = ['valence', 'arousal', 'dominance', 'liking']
label_df = pd.read_csv('./CWT/label.txt',
    usecols = [i for i in range(4)], header=None, delimiter=',' )
print("label_df.shape: ", label_df.shape)
label_df.columns = cols
# (0~10값) 5를 기준으로 작으면 0, 크면 1
label_df[label_df<5] = 0
label_df[label_df>=5] = 1

# valence
label = label_df['arousal'].astype(int).values
label = np.tile(label,20)
print("label.shape: ", label.shape)

# divive train & test
#test_size -> 트레인 vs 테스트 비율 (9:1)
x_train, x_test, y_train, y_test = train_test_split(reshape_dfs, label, test_size=0.1, random_state=1)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()

n = x_train.shape[0]
print(n)

class cnn_classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv11 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.conv12 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.pool1 = nn.MaxPool3d(kernel_size=2, padding=(0,0,1))
    
    self.conv21 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.pool2 = nn.MaxPool3d(kernel_size=2, padding=(0,0,1))
    
    self.conv31 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.conv32 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.pool3 = nn.MaxPool3d(kernel_size=2, padding=0)
    

    self.fc_layer = nn.Linear(128*4*4*1, 2)
    
    self.dropout_layer = nn.Dropout(p=0.5)

  def forward(self, xb):
    h1 = self.conv11(xb)
    h1 = self.conv12(h1)
    h1 = self.dropout_layer(h1)
    h1 = self.pool1(h1)
    h1 = F.relu(h1)

    h2 = self.conv21(h1)
    h2 = self.conv22(h2)
    #h2 = self.dropout_layer(h2)
    h2 = self.pool2(h2)
    h2 = F.relu(h2) 

    h3 = self.conv31(h2)
    h3 = self.conv32(h3)
    #h3 = self.dropout_layer(h3)
    h3 = self.pool3(h3)
    h3 = F.relu(h3) 
    
    
    # flatten the output from conv layers before feeind it to FC layer
    flatten = h3.view(-1, 128*4*4*1)
    out = self.fc_layer(flatten)
    #out = self.dropout_layer(out)
    return out

def train_model(model, x_train, y_train, x_test, y_test, epochs=50 , batch_size=32, lr=0.0001, weight_decay=0):
  # data
  train_dataset = TensorDataset(x_train, y_train)
  train_data_loader = DataLoader(train_dataset, batch_size=batch_size)

  # loss function
  loss_func = F.cross_entropy

  # optimizer
  #optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  # figure
  train_a = list([])
  test_a = list([])

  # training loop
  for epoch in range(epochs):
    model.train()
    tic = time.time()
    acc_train = []
    for xb, yb in train_data_loader:    
      xb, yb = xb.to(device), yb.to(device)
      pred = model(xb)
      loss = loss_func(pred, yb)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      acc_train.append(pred.detach().argmax(1).eq(yb).float().mean().cpu().numpy())
    acc_train = np.mean(acc_train)
    toc = time.time()
    
    with torch.no_grad():
      model.eval()
      y_pred = model(x_test.to(device))
      acc = y_pred.argmax(1).eq(y_test.to(device)).float().mean().cpu().numpy()

    train_a.append(acc_train)
    test_a.append(acc)
    print('Loss at epoch %d : %f, train_acc: %f, test_acc: %f, running time: %d'% (epoch, loss.item(), acc_train, acc, toc-tic))

  # draw an accuray figure
  plt.plot(train_a,'y.-.')
  plt.plot(test_a,'.-.')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')

model = cnn_classifier()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
train_model(model, x_train.view(-1, 1, 32, 32, depth), y_train, x_test.view(-1, 1, 32, 32, depth), y_test)