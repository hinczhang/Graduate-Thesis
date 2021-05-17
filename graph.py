#展示Sigmoid和ReLu
import math 
import numpy as np
import matplotlib.pyplot as plt
 
# set x's range
x = np.arange(-10,10,0.1)
 
y1=1/(1+math.e**(-x)) # sigmoid
#y2=math.e**(-x)/((1+math.e**(-x))**2)
y2=(math.e**(x)-math.e**(-x))/(math.e**(x)+math.e**(-x)) # tanh
y3=np.where(x<0,0,x) # relu
 
plt.xlim(-4,4)
plt.ylim(-1,1)
 
ax = plt.gca()
ax.spines['right'].set_color('none') 
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')   
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
 
#Draw pic
plt.plot(x,y1,label='sigmoid',linestyle="-", color="red")
#plt.plot(x,y2,label='tanh',linestyle="-", color="green")
plt.plot(x,y3,label='relu',linestyle="-", color="blue")
 
# Title
plt.legend(['Sigmoid','Relu'])
 
# save pic
plt.savefig('relusigmoid.svg', dpi=600)
 
# show it!!
plt.show()

plt.subplot(1,2,2)
# 绘制sigmoid 函数
fig =  plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
x = np.linspace(-10,10)
y = 1/(1+np.exp(-x)) # 小于0输出0，大于0输出y
 
ax = plt.gca() # 获得当前axis坐标轴对象
ax.spines['right'].set_color('none') # 去除右边界线
ax.spines['top'].set_color('none') # 去除上边界线
 
ax.xaxis.set_ticks_position('bottom') # 指定下边的边作为x轴
ax.yaxis.set_ticks_position('left') # 指定左边的边为y轴
 
ax.spines['bottom'].set_position(('data',0)) # 指定data 设置的bottom（也就是指定的x轴）绑定到y轴的0这个点上
ax.spines['left'].set_position(('data',0))  # 指定y轴绑定到x轴的0这个点上
 
plt.plot(x,y,label = 'ReLU',linestyle='-',color='r')
plt.legend(['sigmoid '])
