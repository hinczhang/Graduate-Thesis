%matplotlib inline
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
torch.__version__

TIME_STEP = 10 # rnn 时序步长数
INPUT_SIZE = 9 # rnn 的输入维度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
H_SIZE = 64 # of rnn 隐藏单元个数
EPOCHS=300 # 总共训练次数
h_state = None # 隐藏层状态

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
        input_size=INPUT_SIZE,
        hidden_size=H_SIZE, 
        num_layers=1, 
        batch_first=True,
        )
        self.out = nn.Linear(H_SIZE, 1)
    def forward(self, x, h_state):
         # x (batch, time_step, input_size)
         # h_state (n_layers, batch, hidden_size)
         # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = [] # 保存所有的预测值
        for time_step in range(r_out.size(1)): # 计算每一步长的预测值
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
         # 也可使用以下这样的返回值
         # r_out = r_out.view(-1, 32)
         # outs = self.out(r_out)
         # return outs, h_state

rnn = RNN().to(DEVICE)
optimizer = torch.optim.Adam(rnn.parameters()) # Adam优化，几乎不用调参
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差

from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
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

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
Y = data['lines'].values.astype('float32').reshape(-1, 1)
Y = y_scaler.fit_transform(Y)
X = data.drop(columns=result_var).values.astype('float32').reshape(-1, 1)
X = x_scaler.fit_transform(X)

X = torch.tensor(X,dtype=torch.float32).reshape(1995,10,9)
Y = torch.tensor(Y,dtype=torch.float32).reshape(1995,10,1)

testX=X[:500]
testY=Y[:500]
X=X[501:]
Y=Y[501:]

from torch.autograd import Variable
import math

#rnn.train()
h_state = None
i=0
record=[]
for step in range(EPOCHS):
    i=i+1
    #保证scalar类型为Double
    rnn = rnn.double()
    X=X.double().to(DEVICE)
    prediction, h_state = rnn(X, h_state) # rnn output
    # 这一步非常重要
    h_state = h_state.data # 重置隐藏层的状态, 切断和前一次迭代的链接
    loss = criterion(prediction.cpu(), Y.double()) 
    record.append(loss.cpu().data.item())
    # 这三行写在一起就可以
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

#test
h_state = None
prediction, h_state = rnn(testX.double().to(DEVICE), h_state)

y1 = np.array(testY).flatten().tolist()
y2=prediction.data.cpu().numpy().flatten()

loss = criterion(prediction, testY.double().to(DEVICE)) 
print(loss)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False

fmri=pd.DataFrame({"训练次数":list(range(1,len(record)+1)),"MSE标准差":np.sqrt(record)})
sns.lineplot(x="训练次数", y="MSE标准差", markers="o", data=fmri)
plt.savefig("RNN训练.svg",dpi=600)
plt.show()