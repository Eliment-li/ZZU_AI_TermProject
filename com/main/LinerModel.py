import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# 使用全连接神经网络，对人脸图像进行分类
class TrainDataSet(Dataset):

    def __init__(self, filePath):
        file = pd.read_csv(filePath)
        xdata = file.iloc[1:28000, 1].values

        x = []
        for i in range(len(xdata)):
            # cvs文件中，图像数据被连起来放在第一列中，这里将数据分解成数组的形式
            temp = xdata[i].split()
            x.append(temp)

        Scaler = StandardScaler();
        xTrain = Scaler.fit_transform(x)

        y = file.iloc[1:28000, 0].values
        yTrain = y

        self.xTrain = torch.tensor(xTrain, dtype=torch.float32)
        self.yTrain = torch.tensor(yTrain)

    def __len__(self):
        return len(self.yTrain)

    def __getitem__(self, item):
        return self.xTrain[item], self.yTrain[item]


class TestDataSet(Dataset):

    def __init__(self, filePath):
        file = pd.read_csv(filePath)
        xdata = file.iloc[28000:28300, 1].values

        x = []
        for i in range(len(xdata)):
            # cvs文件中，图像数据被连起来放在第一列中，这里将数据分解成数组的形式
            temp = xdata[i].split()
            x.append(temp)

        Scaler = StandardScaler();
        xTrain = Scaler.fit_transform(x)

        y = file.iloc[28000:28300, 0].values
        yTrain = y

        self.xTrain = torch.tensor(xTrain, dtype=torch.float32)
        self.yTrain = torch.tensor(yTrain)

    def __len__(self):
        return len(self.yTrain)

    def __getitem__(self, item):
        return self.xTrain[item], self.yTrain[item]


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(2304, 1500)
        self.l2 = torch.nn.Linear(1500, 800)
        self.l3 = torch.nn.Linear(800, 400)
        self.l4 = torch.nn.Linear(400, 200)
        self.l5 = torch.nn.Linear(200, 100)
        self.l6 = torch.nn.Linear(100, 7)

    def forward(self, x):
        x = x.view(-1, 2304)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)


model = Net()

trainDataSet = TrainDataSet(r'd:\project2_train.csv')
testDataSet = TestDataSet(r'd:\project2_train.csv')

trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=30, shuffle=True)
testLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=30, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    runninLoss = 0.0

    for batch_idx, data in enumerate(trainLoader, 0):
        inputs, target = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        runninLoss += loss.item()

    # 每500个样本输出一次精度
    if batch_idx % 500 == 0:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, runninLoss / 500))
        runninLoss = 0.0


def test():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testLoader:
            images, labels = data

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':

    for epoch in range(10):
        train(epoch)
        test()