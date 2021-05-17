#线性层学习Gdelt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np
data=pd.read_csv("./output.csv")
#训练结果
result_var='lines'
#分类型数据
cat_names=['flow']
#数值型数据
cont_names=['QuadClass', 'GoldsteinScale', 'NumMentions','NumSources', 'NumArticles', 'AvgTone', 'date']

#分类型数据转成数字型
for col in data.columns:
    if col in cat_names:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))
Y = data['lines']
X = data.drop(columns=result_var)

class tabularDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
train_ds = tabularDataset(X, Y)
#训练前指定使用的设备
DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda")
print(DEVICE)
#损失函数
criterion = torch.nn.MSELoss()

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(9, 500)
        self.lin2 = nn.Linear(500, 100)
        self.lin3 = nn.Linear(100, 1)
        self.bn0 = nn.BatchNorm1d(9)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(100)
        

    def forward(self,x_in):
        #print(x_in.shape)
        x = self.bn0(x_in)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        #print(x)
        
        
        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        #print(x)
        
        x = self.lin3(x)
        x=torch.sigmoid(x)
        x = x.squeeze(-1)
        return x

class MoreCellModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(9, 1000)
        self.lin2 = nn.Linear(1000, 100,3)
        self.lin3 = nn.Linear(100, 1)
        self.test=nn.Softmax(1)
        self.bn0 = nn.BatchNorm1d(9)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(100)
        

    def forward(self,x_in):
        #print(x_in.shape)
        x = self.bn0(x_in)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        #print(x)
        x=self.test(x)
        
        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        #print(x)
        
        x = self.lin3(x)
        x=torch.sigmoid(x)
        x = x.squeeze(-1)
        return x

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(9, 500)
        self.lin2 = nn.Linear(500, 1000)
        self.lin3 = nn.Linear(1000, 200)
        self.lin4 = nn.Linear(200, 100)
        self.lin5 = nn.Linear(100, 1)
        self.bn0 = nn.BatchNorm1d(9)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(200)
        self.bn4 = nn.BatchNorm1d(100)

    def forward(self,x_in):
        x = self.bn0(x_in)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        x = F.relu(self.lin3(x))
        x = self.bn3(x)
        x = F.relu(self.lin4(x))
        x = self.bn4(x)
        x=torch.sigmoid(self.lin5(x))
        x = x.squeeze(-1)
        return x

#实例化模型
model1 = LinearModel().to(DEVICE)
model2 = MoreCellModel().to(DEVICE)
model3 = ComplexModel().to(DEVICE)
#学习率
LEARNING_RATE=0.2
#BS
batch_size = 1024
#优化器
optimizer1 = torch.optim.Adam(model1.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))
optimizer2 = torch.optim.Adam(model2.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))
optimizer3 = torch.optim.Adam(model3.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))

#测试模型是否没问题
rn1=torch.rand(3,9).to(DEVICE)
rn2=torch.rand(3,9).to(DEVICE)
rn3=torch.rand(3,9).to(DEVICE)
print(model1(rn1),model2(rn2),model3(rn3))

#DataLoader加载数据
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
train_dl

from torch.autograd import Variable
import math
#%%time
model1.train()
#训练10轮
TOTAL_EPOCHS=100
#记录损失函数
losses = []
record1=[]
for epoch in range(TOTAL_EPOCHS):
    for i, (x, y) in enumerate(train_dl):
        x = Variable(x.float().to(DEVICE)) #输入必须未float类型
        y = Variable(y.float().to(DEVICE))#结果标签必须未long类型
        
        #清零
        optimizer1.zero_grad()
        outputs = model1(x)
        #计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer1.step()
        losses.append(loss.cpu().data.item())
    #print(losses)
    print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, math.sqrt(np.mean(losses))))
    record1.append(math.sqrt(np.mean(losses)))

from torch.autograd import Variable
import math
#%%time
model2.train()
#训练10轮
TOTAL_EPOCHS=100
#记录损失函数
losses = []
record2=[]
for epoch in range(TOTAL_EPOCHS):
    for i, (x, y) in enumerate(train_dl):
        x = Variable(x.float().to(DEVICE)) #输入必须未float类型
        y = Variable(y.float().to(DEVICE))#结果标签必须未long类型
        
        #清零
        optimizer2.zero_grad()
        outputs = model2(x)
        #计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer2.step()
        losses.append(loss.cpu().data.item())
    #print(losses)
    print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, math.sqrt(np.mean(losses))))
    record2.append(math.sqrt(np.mean(losses)))

from torch.autograd import Variable
import math
#%%time
model3.train()
#训练10轮
TOTAL_EPOCHS=100
#记录损失函数
losses = []
record3=[]
for epoch in range(TOTAL_EPOCHS):
    for i, (x, y) in enumerate(train_dl):
        x = Variable(x.float().to(DEVICE)) #输入必须未float类型
        y = Variable(y.float().to(DEVICE))#结果标签必须未long类型
        
        #清零
        optimizer3.zero_grad()
        outputs = model3(x)
        #计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer3.step()
        losses.append(loss.cpu().data.item())
    #print(losses)
    print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, math.sqrt(np.mean(losses))))
    record3.append(math.sqrt(np.mean(losses)))

showCase=pd.DataFrame({"训练次数":list(range(1,TOTAL_EPOCHS+1)),"简单ReLu激活网络":record1,"多单元网络":record2,"多层网络":record3})

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

wide_df = pd.DataFrame(showCase[["简单ReLu激活网络","多单元网络","多层网络"]], showCase["训练次数"], ["简单ReLu激活网络","多单元网络","多层网络"])
ax = sns.lineplot(data=wide_df)
plt.savefig("Relu训练.svg",dpi=600)
plt.show()