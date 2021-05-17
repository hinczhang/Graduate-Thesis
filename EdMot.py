#社区划分并导出文件
import networkx as nx
import pandas as pd
from karateclub import EdMot
file="202012"
data=pd.read_csv("./weighted/"+file+".csv",low_memory=False)
G=nx.from_pandas_edgelist(data,"org_code","dst_code",True,nx.Graph)
splitter = EdMot()
splitter.fit(G)
member=splitter.get_memberships()
n_member=[[] for item in range(max(member.values())+1)]
for i in member.keys():
    n_member[member[i]].append(i)
n_member=sorted(n_member,key=len,reverse=True)

airports={}
for index, item in data.iterrows():
    airports[item["org_code"]]={"lat":item["org_lat"],"lon":item["org_lon"],"name":item["org_name"],"city":item["org_city"],"country":item["org_country"],"continent":item["org_continent"]}
    airports[item["dst_code"]]={"lat":item["dst_lat"],"lon":item["dst_lon"],"name":item["dst_name"],"city":item["dst_city"],"country":item["dst_country"],"continent":item["dst_continent"]}
n_member=n_member[:7]
output_data=[]
for index,item in enumerate(n_member):
    for code in item:
        output_data.append({"air_code":code,"lat":airports[code]["lat"],"lon":airports[code]["lon"],
                            "city":airports[code]["city"],"country":airports[code]["country"],
                            "continent":airports[code]["continent"],"group":index})
output_data=pd.DataFrame(output_data)
output_data.to_csv("./community/"+file+".csv")

#绘制社区划分后社区规模分布图
import networkx as nx
import pandas as pd
from karateclub import EdMot
date=list(range(201901,201913))
date+=list(range(202001,202013))
record=[]
for day in date:
    file=str(day)
    data=pd.read_csv("./weighted/"+file+".csv",low_memory=False)
    G=nx.from_pandas_edgelist(data,"org_code","dst_code",True,nx.Graph)
    splitter = EdMot()
    splitter.fit(G)
    member=splitter.get_memberships()
    n_member=[[] for item in range(max(member.values())+1)]
    for i in member.keys():
        n_member[member[i]].append(i)
    n_member=sorted(n_member,key=len,reverse=True)
    record.append(list(map(lambda x:len(x),n_member)))
    
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
plt.boxplot(record,showmeans=True,patch_artist=True,
            boxprops = {'color':'black','facecolor':'#9999ff'},
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
            meanprops = {'marker':'D','markerfacecolor':'indianred'},
            medianprops = {'linestyle':'--','color':'orange'})
plt.tick_params(top='off', right='off')
plt.ylabel('集团规模')
plt.xlabel('月份（序列化）')
plt.title("航空网络聚集状况箱型图")
plt.savefig("航空网络聚集状态.png",dpi = 600,bbox_inches='tight')
plt.show()