#度和热度计算
import pandas as pd
import networkx as nx
date=list(range(201901,201913))
date+=list(range(202001,202013))

for day in date:
    data=pd.read_csv("./month/"+str(day)+".csv",low_memory=False)
    data=data[(data["org_continent"]=="EU")&(data["dst_continent"]=="EU")]
    G=nx.from_pandas_edgelist(data,"origin","destination",True,nx.MultiDiGraph)
    airports={}
    for index, item in data.iterrows():
        airports[item["origin"]]={"lat":item["org_lat"],"lon":item["org_lon"],"name":item["org_name"],"city":item["org_city"],"country":item["org_country"],"continent":item["org_continent"]}
        airports[item["destination"]]={"lat":item["dst_lat"],"lon":item["dst_lon"],"name":item["dst_name"],"city":item["dst_city"],"country":item["dst_country"],"continent":item["dst_continent"]}
    degree=G.degree()
    output=[]
    for item in degree:
        output.append({"name":item[0],"lat":airports[item[0]]["lat"],"lon":airports[item[0]]["lon"],
                       "city":airports[item[0]]["city"],"country":airports[item[0]]["country"],
                       "continent":airports[item[0]]["continent"],"weight":item[1]})
    pd.DataFrame(output).to_csv("./hot/"+str(day)+".csv")
    print(day)
#图的案例绘制
import networkx as nx
import matplotlib.pyplot as plt
import random
plt.figure(dpi=600,figsize=(15,12))

plt.subplot(2,2,1)
#啥都没
G=nx.Graph()
for u, v in nx.barabasi_albert_graph(10,2,seed=1).edges():
    G.add_edge(u,v)
pos=nx.spring_layout(G,iterations=20)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_nodes(G,pos)
plt.title("(a)", y=-0.15,fontsize = 24)
plt.subplot(2,2,2)
#方向
G=nx.DiGraph()
for u, v in nx.barabasi_albert_graph(10,2,seed=1).edges():
    G.add_edge(u,v)
pos=nx.spring_layout(G,iterations=20)
#以下语句绘制以带宽为线的宽度的图
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_nodes(G,pos)
plt.title("(b)", y=-0.15,fontsize = 24)
plt.subplot(2,2,3)
#权重
G=nx.Graph()
for u, v in nx.barabasi_albert_graph(10,2,seed=1).edges():
    G.add_edge(u,v,weight=random.uniform(0,0.4))
pos=nx.spring_layout(G,iterations=20)
#以下语句绘制以带宽为线的宽度的图
nx.draw_networkx_edges(G,pos,width=[float(d['weight']*10) for (u,v,d) in G.edges(data=True)])
nx.draw_networkx_nodes(G,pos)
plt.title("(c)", y=-0.15,fontsize = 24)
plt.subplot(2,2,4)
#方向权重
G=nx.DiGraph()
for u, v in nx.barabasi_albert_graph(10,2,seed=1).edges():
    G.add_edge(u,v,weight=random.uniform(0,0.4))
pos=nx.spring_layout(G,iterations=20)
#以下语句绘制以带宽为线的宽度的图
nx.draw_networkx_edges(G,pos,width=[float(d['weight']*10) for (u,v,d) in G.edges(data=True)])
nx.draw_networkx_nodes(G,pos)
plt.title("(d)", y=-0.15,fontsize = 24)
plt.savefig('./graph.svg', dpi=600, bbox_inches='tight')
plt.show()