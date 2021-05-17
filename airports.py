#对于首位机场排序进行计算
import pandas as pd
import networkx as nx
date=list(range(201901,201913))
date+=list(range(202001,202013))

def for_degree(degree):
    degree=sorted(degree,key=lambda x:x[1],reverse=True)
    return degree[:5]

def for_others(array):
    temp=[]
    for i in array.keys():
        temp.append((i,array[i]))
    temp=sorted(temp,key=lambda x:x[1],reverse=True)
    return temp[:5]

degree=[]
cent=[]
page=[]
auts=[]
hubs=[]
for day in date:
    data=pd.read_csv("./month/"+str(day)+".csv",low_memory=False)
    data=data[(data["org_continent"]=="EU")&(data["dst_continent"]=="EU")]
    G=nx.from_pandas_edgelist(data,"origin","destination",True,nx.MultiDiGraph)
    data=pd.read_csv("./weighted/"+str(day)+".csv",low_memory=False)
    data=data[(data["org_continent"]=="EU")&(data["dst_continent"]=="EU")]
    WG=nx.from_pandas_edgelist(data,"origin","destination",True,nx.DiGraph)
    
    degree.append(for_degree(G.degree()))
    cent.append(for_others(nx.algorithms.centrality.degree_centrality(G)))
    page.append(for_others(nx.algorithms.link_analysis.pagerank_alg.pagerank(WG)))
    auts.append(for_others(nx.algorithms.link_analysis.hits_alg.hits(WG)[0]))
    hubs.append(for_others(nx.algorithms.link_analysis.hits_alg.hits(WG)[1]))
    
    print(day)

import squarify
import matplotlib.pyplot as plt
import seaborn as sns

date=list(range(201901,201913))
date+=list(range(202001,202013))
plt.figure(dpi=600,figsize=(16,12))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#plt.title("度指数排名前五的机场")
colors = sns.color_palette(palette="bright",n_colors=5)
def draw_pic(index,day,data): 
    plt.subplot(4,6,index)
    labels=list(map(lambda x:x[0]+"\n("+str(x[1])+")",data))
    sizes=list(map(lambda x:x[1],data))
    squarify.plot(sizes=sizes,label=labels, color=colors, alpha=.8)
    plt.title(str(day),y=-0.15)
    plt.axis('off')
for index,day in enumerate(date):
    draw_pic(index+1,day,degree[index])
plt.savefig('./degree_port.svg', dpi=600, bbox_inches='tight')
plt.show()
degree_map=[]
for i in degree:
    for j in i:
        degree_map.append(j[0])
from collections import Counter
import matplotlib.pyplot as plt
plt.figure(dpi=600,figsize=(8,6))
sns.set_style('darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
tu=(dict(Counter(degree_map)))
tu_x=list(tu.keys())
tu_y=list(map(lambda x:tu[x],tu_x))
sns.barplot(tu_x,tu_y,palette=sns.color_palette('YlGn'))
plt.title("度指数前五的机场出现频率")
plt.savefig('./Degree_freq.svg', dpi=600, bbox_inches='tight')
plt.show()

import squarify
import matplotlib.pyplot as plt
import seaborn as sns

date=list(range(201901,201913))
date+=list(range(202001,202013))
plt.figure(dpi=600,figsize=(16,12))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#plt.title("度指数排名前五的机场")
colors = sns.color_palette(palette="deep",n_colors=5)
def draw_pic(index,day,data): 
    plt.subplot(4,6,index)
    labels=list(map(lambda x:x[0]+"\n("+str(('%.4f' % x[1]))+")",data))
    sizes=list(map(lambda x:x[1],data))
    squarify.plot(sizes=sizes,label=labels, color=colors, alpha=.8)
    plt.title(str(day),y=-0.15)
    plt.axis('off')
for index,day in enumerate(date):
    draw_pic(index+1,day,cent[index])
plt.savefig('./cent_port.svg', dpi=600, bbox_inches='tight')
plt.show()
degree_map=[]
for i in cent:
    for j in i:
        degree_map.append(j[0])
from collections import Counter
import matplotlib.pyplot as plt
plt.figure(dpi=600,figsize=(8,6))
sns.set_style('darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
tu=(dict(Counter(degree_map)))
tu_x=list(tu.keys())
tu_y=list(map(lambda x:tu[x],tu_x))
sns.barplot(tu_x,tu_y,palette=sns.color_palette('Blues_r'))
plt.title("阶数中心性前五的机场出现频率")
plt.savefig('./Cent_freq.svg', dpi=600, bbox_inches='tight')
plt.show()

import squarify
import matplotlib.pyplot as plt
import seaborn as sns

date=list(range(201901,201913))
date+=list(range(202001,202013))
plt.figure(dpi=600,figsize=(16,12))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#plt.title("度指数排名前五的机场")
colors = sns.color_palette(palette="muted",n_colors=5)
def draw_pic(index,day,data): 
    plt.subplot(4,6,index)
    labels=list(map(lambda x:x[0]+"\n("+str(('%.4f' % x[1]))+")",data))
    sizes=list(map(lambda x:x[1],data))
    squarify.plot(sizes=sizes,label=labels, color=colors, alpha=.8)
    plt.title(str(day),y=-0.15)
    plt.axis('off')
for index,day in enumerate(date):
    draw_pic(index+1,day,page[index])
plt.savefig('./page_port.svg', dpi=600, bbox_inches='tight')
plt.show()
degree_map=[]
for i in page:
    for j in i:
        degree_map.append(j[0])
from collections import Counter
import matplotlib.pyplot as plt
plt.figure(dpi=600,figsize=(10,6))
sns.set_style('darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
tu=(dict(Counter(degree_map)))
tu_x=list(tu.keys())
tu_y=list(map(lambda x:tu[x],tu_x))
sns.barplot(tu_x,tu_y,palette=sns.color_palette("ch:s=.25,rot=-.25"))
plt.title("PageRank前五的机场出现频率")
plt.savefig('./Page_freq.svg', dpi=600, bbox_inches='tight')
plt.show()

import squarify
import matplotlib.pyplot as plt
import seaborn as sns

date=list(range(201901,201913))
date+=list(range(202001,202013))
plt.figure(dpi=600,figsize=(16,12))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#plt.title("度指数排名前五的机场")
colors = sns.color_palette(palette="pastel",n_colors=5)
def draw_pic(index,day,data): 
    plt.subplot(4,6,index)
    labels=list(map(lambda x:x[0]+"\n("+str(('%.4f' % x[1]))+")",data))
    sizes=list(map(lambda x:x[1],data))
    squarify.plot(sizes=sizes,label=labels, color=colors, alpha=.8)
    plt.title(str(day),y=-0.15)
    plt.axis('off')
for index,day in enumerate(date):
    draw_pic(index+1,day,auts[index])
plt.savefig('./auts_port.svg', dpi=600, bbox_inches='tight')
plt.show()
degree_map=[]
for i in auts:
    for j in i:
        degree_map.append(j[0])
from collections import Counter
import matplotlib.pyplot as plt
plt.figure(dpi=600,figsize=(10,6))
sns.set_style('darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
tu=(dict(Counter(degree_map)))
tu_x=list(tu.keys())
tu_y=list(map(lambda x:tu[x],tu_x))
sns.barplot(tu_x,tu_y,palette=sns.color_palette("BuPu_r"))
plt.title("Authority前五的机场出现频率")
plt.savefig('./Auts_freq.svg', dpi=600, bbox_inches='tight')
plt.show()

import squarify
import matplotlib.pyplot as plt
import seaborn as sns

date=list(range(201901,201913))
date+=list(range(202001,202013))
plt.figure(dpi=600,figsize=(16,12))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#plt.title("度指数排名前五的机场")
colors = sns.color_palette(palette="colorblind",n_colors=5)
def draw_pic(index,day,data): 
    plt.subplot(4,6,index)
    labels=list(map(lambda x:x[0]+"\n("+str(('%.4f' % x[1]))+")",data))
    sizes=list(map(lambda x:x[1],data))
    squarify.plot(sizes=sizes,label=labels, color=colors, alpha=.8)
    plt.title(str(day),y=-0.15)
    plt.axis('off')
for index,day in enumerate(date):
    draw_pic(index+1,day,hubs[index])
plt.savefig('./hubs_port.svg', dpi=600, bbox_inches='tight')
plt.show()
degree_map=[]
for i in hubs:
    for j in i:
        degree_map.append(j[0])
from collections import Counter
import matplotlib.pyplot as plt
plt.figure(dpi=600,figsize=(10,6))
sns.set_style('darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
tu=(dict(Counter(degree_map)))
tu_x=list(tu.keys())
tu_y=list(map(lambda x:tu[x],tu_x))
sns.barplot(tu_x,tu_y,palette=sns.color_palette("magma_r"))
plt.title("Hub前五的机场出现频率")
plt.savefig('./Hubs_freq.svg', dpi=600, bbox_inches='tight')
plt.show()