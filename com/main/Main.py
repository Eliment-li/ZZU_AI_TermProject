import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# transform = transforms.Compose([
# transforms.ToTensor(),
# transforms.Normalize((0.1307, ), (0.3081, ))
# ])

class StanderDataSet(Dataset):
    def __init__(self,filePath):
        file=pd.read_csv(filePath)
        xdata=file.iloc[1:200,1].values
        x=[]
        for i in range(len(xdata)):
            temp=xdata[i].split()
            x.append(temp)


        y=file.iloc[1:200,0].values

        Scaler=StandardScaler();
        xTrain=Scaler.fit_transform(x)
        yTrain=y

        self.xTrain=torch.tensor(xTrain,dtype=torch.float32)
        self.yTrain=torch.tensor(yTrain)
    def __len__(self):
        return len(self.yTrain)
    def __getitem__(self, item):
        return self.xTrain[item],self.yTrain[item]

Datas=StanderDataSet(r'd:\project2_train.csv')
trainLoader=torch.utils.data.DataLoader(Datas,batch_size=10,shuffle=True)
