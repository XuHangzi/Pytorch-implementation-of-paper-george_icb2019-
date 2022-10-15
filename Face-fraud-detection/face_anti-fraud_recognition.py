import os

import numpy as np

import torch as t
import torch.nn as nn
from torch.optim import  lr_scheduler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import  models

import matplotlib.pyplot as plt
import cv2

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
#制作datase
class MyDataSet(Dataset):
    '''
    定义数据集，用于将读取到的图片数据转换并处理成CNN神经网络需要的格式
    '''
    def __init__(self, DataArray, LabelArray):
        super(MyDataSet, self).__init__()
        self.data = DataArray
        self.label = LabelArray

    def __getitem__(self, index):
        # 对图片的预处理步骤
        # 1. 中心缩放至224(ResNet的输入大小)
        # 2. 随机旋转0-30°
        # 3. 对图片进行归一化，参数来源为pytorch官方文档
        im_trans = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.CenterCrop(size=224),
            #transforms.RandomRotation((0, 30)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return im_trans(self.data[index]), t.tensor(self.label[index], dtype=t.float32)

    def __len__(self):
        return self.label.shape[0]

# 读取LFW数据集，将图片数据读入数组并将名字转换为标签
path_train_true =  r'./video/train_true'
path_train_false =  r'./video/train_false'
pathlist_train_true = map(lambda x: '\\'.join([path_train_true, x]), os.listdir(path_train_true))
pathlist_train_false = map(lambda x: '\\'.join([path_train_false, x]), os.listdir(path_train_false))
data_train, label_train = [], []
idx_true = np.zeros((7,7),np.float32)#真假标签
idx_false=np.ones((7,7),np.float32)
#print(idx_false)
for path in pathlist_train_true:
        data_train.append(cv2.imread(path))
        label_train.append(idx_true)
for path in pathlist_train_false:
        data_train.append(cv2.imread(path))
        label_train.append(idx_false)
#读取测试数据集，将图片数据读入数组并将写入对应标签
path_test_true=r'./video/test_true'
path_test_false=r'./video/test_false'
pathlist_test_true = map(lambda x: '\\'.join([path_test_true, x]), os.listdir(path_test_true))
pathlist_test_false = map(lambda x: '\\'.join([path_test_false, x]), os.listdir(path_test_false))
data_test, label_test = [], []
#idx_true = 0
#idx_false=1
for path in pathlist_test_true:
        data_test.append(cv2.imread(path))
        label_test.append(idx_true)
for path in pathlist_test_false:
        data_test.append(cv2.imread(path))
        label_test.append(idx_false)
# 随机打乱数据，得到训练集和测试集
data_train, label_train = np.stack(data_train), np.array(label_train)
idx_train = np.random.permutation(data_train.shape[0])
data_train, label_train = data_train[idx_train], label_train[idx_train]

data_test, label_test = np.stack(data_test), np.array(label_test)
idx_test = np.random.permutation(data_test.shape[0])
data_test, label_test = data_test[idx_test], label_test[idx_test]
train_X=data_train;train_Y=label_train;test_X=data_test;test_Y=label_test

# 将分割好的训练集和测试集处理为pytorch所需的格式
TrainSet = MyDataSet(train_X, train_Y)
TestSet = MyDataSet(test_X, test_Y)
TrainLoader = DataLoader(TrainSet, batch_size=32, shuffle=True, drop_last=True)
TestLoader = DataLoader(TestSet, batch_size=32, shuffle=True, drop_last=True)
#迁移学习修改resnet
class MyResnet(nn.Module):
    def __init__(self , model):
        super(MyResnet, self).__init__()
        #取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])        
        self.conv_layer = nn.Conv2d(2048, 1, kernel_size=1, bias=False)
        
        
    def forward(self, x):
        x = self.resnet_layer(x) 
        x=self.conv_layer(x)        
        return x

resnet = models.resnet50(pretrained=True)
resnet=resnet.to(DEVICE)
model = MyResnet(resnet)
model=model.to(DEVICE)
lossf = nn.MSELoss()
LR=0.1
optimizer = t.optim.Adam(model.parameters(),LR)
# 定义一个list保存学习率
lr_list = []

# 定义学习率与轮数关系的函数
lambda1 = lambda epoch:0.95 ** epoch # 学习率 = 0.95**(轮数)              
scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)

#进行训练和测试
def main():
    epocs=10
    train_loss_arr, train_acc_arr, test_loss_arr, test_acc_arr = [], [], [], []
    for epoc in range(epocs):
        model.train()
        TrainLoss, TrainAcc = 0, 0
        for BatchIdx, (InputData, Labels) in enumerate(TrainLoader):
            optimizer.zero_grad()
            InputData = InputData.to(DEVICE)
            Labels = Labels.to(DEVICE)
            Outputs = model(InputData)
            loss = lossf(Outputs.squeeze(), Labels)
            loss.backward()
            optimizer.step()
            TrainLoss += loss.item()
            out=t.round(t.mean(t.mean(Outputs.squeeze(),2),1))
            lab=t.round(t.mean(t.mean(Labels,2),1))
            TrainAcc += t.mean(out.eq(lab.data.view_as(out)).type(t.FloatTensor)).item() * len(InputData)
            if BatchIdx % 10 == 0 and BatchIdx > 0:
                print('Bathch: {}/{}\tLoss: {}\tAcc: {}%'.format(BatchIdx, len(TrainLoader), round(TrainLoss, 2), 
                                                            round(100*TrainAcc/((BatchIdx+1) * InputData.shape[0]), 2)))
        train_acc_arr.append(100*TrainAcc/(len(TrainLoader)*32))
        train_loss_arr.append(TrainLoss)
        
        TestLoss, TestAcc = 0, 0
        with t.no_grad():
            model.eval()
            for BatchIdx, (InputData, Labels) in enumerate(TestLoader):
                InputData = InputData.to(DEVICE)
                Labels = Labels.to(DEVICE)
                Outputs = model(InputData)
                loss = lossf(Outputs.squeeze(), Labels)
                TestLoss += loss.item()
                out=t.round(t.mean(t.mean(Outputs.squeeze(),2),1))
                lab=t.round(t.mean(t.mean(Labels,2),1))
                TestAcc += t.mean(out.eq(lab.data.view_as(out)).type(t.FloatTensor)).item() * len(InputData)
            print('Loss: {}\tAcc: {}%'.format(round(TrainLoss, 2),
                                              round(100*TestAcc/(len(TestLoader) * 32), 2)))
            print('-'*60)  
        test_acc_arr.append(100*TestAcc/(len(TestLoader)*32))
        test_loss_arr.append(TestLoss)


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(train_loss_arr, label='train loss')
    ax1.plot(test_loss_arr, label='test loss')
    ax1.legend()
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('epocs')
    ax1.set_ylabel('loss')
    ax2 = fig.add_subplot(122)
    ax2.plot(train_acc_arr, label='train acc')
    ax2.plot(test_acc_arr, label='test acc')
    ax2.legend()
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('epocs')
    ax2.set_ylabel('loss')
    plt.show()
if __name__ == '__main__':
    main()