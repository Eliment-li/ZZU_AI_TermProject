import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import  numpy as np
import torch.optim as optim



class TrainDataSet(Dataset):

    def __init__(self, filePath):
        file = pd.read_csv(filePath)
        xdata = file.iloc[1:28000, 1].values

        x = []
        for i in range(len(xdata)):
            # cvs文件中，图像数据被连起来放在第一列中，这里将数据分解成数组的形式
            temp = xdata[i].split()
            temp = list(map(int, temp))
            temp= np.reshape(temp, (1,48, 48))
            temp = torch.tensor(temp, dtype=torch.float32)
            x.append(temp)

        # Scaler = StandardScaler();
        # xTrain = Scaler.fit_transform(x)
        self.xTrain=x
        self.y = file.iloc[1:28000, 0].values
        self.yTrain = self.y

        # self.xTrain = torch.tensor(xTrain, dtype=torch.float32)
        # self.yTrain = torch.tensor(yTrain)

    def __len__(self):
        return len(self.yTrain)

    def __getitem__(self, item):
        return self.xTrain[item], self.yTrain[item]

class TestDataSet(Dataset):

    def __init__(self, filePath):
        file = pd.read_csv(filePath)
        xdata = file.iloc[28000:, 1].values

        x = []
        for i in range(len(xdata)):
            # cvs文件中，图像数据被连起来放在第一列中，这里将数据分解成数组的形式
            temp = xdata[i].split()
            temp = list(map(int, temp))
            temp= np.reshape(temp, (1,48, 48))
            temp = torch.tensor(temp,dtype=torch.float32)
            x.append(temp)

        #Scaler = StandardScaler();
       # xTrain = Scaler.fit_transform(x)
        self.xTrain=x
        self.y = file.iloc[28000:, 0].values
        self.yTrain = self.y

        # self.xTrain = torch.tensor(xTrain, dtype=torch.float32)
        # self.yTrain = torch.tensor(yTrain)

    def __len__(self):
        return len(self.yTrain)

    def __getitem__(self, item):
        return self.xTrain[item], self.yTrain[item]



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3,stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.pooling = torch.nn.MaxPool2d(2)

        self.l1 = torch.nn.Linear(256*6*6, 4096)
        self.l2 = torch.nn.Linear(4096, 1024)
        self.l3 = torch.nn.Linear(1024, 256)
        self.l4 = torch.nn.Linear(256, 7)

    def forward(self, x):


        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))

        batch_size = x.size(0)
        x = x.view(batch_size, -1) # flatten

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return x

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0)#lr=0.1, momentum=0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
trainDataSet = TrainDataSet(r'd:\project2_train.csv')
testDataSet = TestDataSet(r'd:\project2_train.csv')

trainLoader = torch.utils.data.DataLoader(trainDataSet,  shuffle=True)
testLoader = torch.utils.data.DataLoader(trainDataSet,  shuffle=True)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(trainLoader, 0):
        #print("第"+str(batch_idx)+"\n")
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if (batch_idx % 500) ==0:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 500))
        running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))

if __name__ == '__main__':
    for epoch in range(100):
        print('epoch'+str(epoch))
        train(epoch)
        test()