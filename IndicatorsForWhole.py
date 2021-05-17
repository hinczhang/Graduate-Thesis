#计算网络全局指标
import pandas as pd
import networkx as nx
from networkx.algorithms import approximation
from networkx.algorithms import assortativity
date=list(range(201901,201913))
date+=list(range(202001,202013))
output=[]
for day in date:
    data=pd.read_csv("./month/"+str(day)+".csv",low_memory=False)
    data=data[(data["org_continent"]=="EU")&(data["dst_continent"]=="EU")]
    wdata=pd.read_csv("./weighted/"+str(day)+".csv",low_memory=False)
    G=nx.from_pandas_edgelist(data,"origin","destination",True,nx.MultiDiGraph)
    noMultiG=nx.from_pandas_edgelist(data,"origin","destination",True,nx.DiGraph)
    WeightedG=nx.from_pandas_edgelist(wdata,"origin","destination","weight",nx.DiGraph)
    output.append({
        "date":day,
        "avg_cluster_q":approximation.average_clustering(G.to_undirected()),
        "clique_size":approximation.large_clique_size(noMultiG.to_undirected()),
        "degree_assort":assortativity.degree_assortativity_coefficient(WeightedG,weight="weight"),
        "degree_assort_p":assortativity.degree_pearson_correlation_coefficient(WeightedG,weight="weight"),
        "transitivity":nx.transitivity(noMultiG),
        "avg_cluster":nx.average_clustering(WeightedG,weight="weight"),
        "str_con":nx.algorithms.components.number_strongly_connected_components(G),
        "wea_con":nx.algorithms.components.number_weakly_connected_components(G),
        "bin_con":len(list(nx.algorithms.components.biconnected_components(G.to_undirected()))),
        "efficiency":nx.algorithms.efficiency_measures.global_efficiency(G.to_undirected()),
    })
    print(day)
pd.DataFrame(output).to_csv("./wholeMap.csv")

#绘制相关性热力图
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("wholeMap.csv")
df=df.drop(columns = ['Unnamed: 0','date'])
dfData = df.corr()
plt.subplots(figsize=(9, 9)) # 设置画面大小
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
plt.xticks(rotation=60)
plt.rcParams['savefig.dpi'] =600
plt.savefig('./CorrelationRelation.svg', dpi=600, bbox_inches='tight')
plt.show()

#绘制时序图
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator
z_scaler = lambda x : (x - min(x)) / (max(x)-min(x))
df=pd.read_csv("wholeMap.csv")
df[[ 'avg_cluster_q', 'clique_size', 'degree_assort','degree_assort_p', 
    'transitivity', 'avg_cluster', 'str_con', 'wea_con','bin_con', 'efficiency']]=df[[ 'avg_cluster_q', 'clique_size', 'degree_assort','degree_assort_p', 
    'transitivity', 'avg_cluster', 'str_con', 'wea_con','bin_con', 'efficiency']].apply(z_scaler)
df["date"]=df["date"].apply(lambda x:str(x)[2:4]+"-"+str(x)[4:])
plt.figure(dpi=600,figsize=(32,15))
palette=sns.color_palette(palette="dark", n_colors = 10)
names={'avg_cluster_q':'平均聚集系数（启发式）', 'clique_size':'集团规模(启发式)', 'degree_assort':'度同配性','degree_assort_p':'皮尔逊度同配性', 
    'transitivity':'传递性', 'avg_cluster':'平均聚集系数', 'str_con':'强连通成分量', 'wea_con':'弱连通成分量','bin_con':'双连通分量', 'efficiency':'网络效率'}
from scipy.interpolate import make_interp_spline
import numpy as np
sns.set(style='darkgrid')
def drawFig(x,y,data,sub):
    plt.subplot(2,5,sub)
    plt.tick_params(labelsize=15)
    plt.xlabel('月份',fontsize=18)
    plt.ylabel('指数',fontsize=18)
    x_major_locator=MultipleLocator(5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    
    plt.scatter(data[x],data[y],color=palette[sub-1])
    xnew = np.linspace(0,len(data[x])-1,300) #300 represents number of points to make between T.min and T.max
    power_smooth = make_interp_spline(np.array(list(range(len(data[x])))),np.array(list(data[y])))(xnew)
    plt.plot(xnew,power_smooth)
    plt.title(names[y],fontsize=18) 
    

for index, item in enumerate([ 'avg_cluster_q', 'clique_size', 'degree_assort','degree_assort_p', 'transitivity', 'avg_cluster', 'str_con', 'wea_con','bin_con', 'efficiency']):
    drawFig("date",item,df,index+1)
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签 
plt.rcParams['axes.unicode_minus']=False
plt.savefig("Dynamic.svg")
plt.show()


