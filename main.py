import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_samples = 65536
batch_size = 4
batch_count = int(train_samples / batch_size)
epoch_count = 20
eval_size = 1024

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.L1 = nn.Linear(2,20).to(device)
        self.L2 = nn.Linear(20,1).to(device)
    
    def forward(self, x):
        x = F.relu(self.L1(x))
        x = torch.sigmoid(self.L2(x))
        return x


def createDataWithLabels(size: int):
    data = torch.rand(size, 2)
    data = (data * 2 - 1) * ((np.pi / 2)**0.5)
    labels = torch.sqrt(data[:,0]*data[:,0] + data[:,1]*data[:,1])
    labels = torch.heaviside( 1. - labels, torch.ones(labels.shape[0]))

    data = data.to(device)
    labels = labels.to(device)
    
    return data, labels


if __name__ == "__main__":
    print('Welcome!')

    net = Net()
    optim = torch.optim.SGD(net.parameters(), 0.025)
    criterion = nn.MSELoss()
    print(net)

    dataTrain, labelsTrain = createDataWithLabels(batch_count * batch_size)
    dataEval, labelsEval = createDataWithLabels(eval_size)
    onesEval = torch.ones(eval_size).to(device)

    print(f'In: {dataTrain}')
    print(f'Labels: {labelsTrain}')

    timeStart = time.perf_counter()

    for epoch in range(epoch_count):
        timeEpochStart = time.perf_counter()
        # Train the Net
        trainLoss = 0.0
        for bIdx in range(1, batch_count*batch_size, batch_size):
            batch = dataTrain[bIdx:bIdx + batch_size]

            output = net(batch)

            optim.zero_grad()
            loss = criterion(output.squeeze(), labelsTrain[bIdx:bIdx + batch_size])
            loss.backward()
            optim.step()
            trainLoss += loss.item()
        
        timeEpochFinish = time.perf_counter()
        
        # Eval the Net
        pred = torch.heaviside(net(dataEval).squeeze() - 0.5, onesEval)
        evalLoss = criterion(pred, labelsEval)

        timeEval = time.perf_counter()

        print(f'Epoch {epoch + 1}, train loss: {trainLoss / batch_count}, eval loss: {evalLoss}, epoch time: {timeEpochFinish - timeEpochStart}, eval time: {timeEval - timeEpochFinish}')

    timeFinish = time.perf_counter()

    print(f'Total time: {timeFinish - timeStart}')

    print('Done!')
