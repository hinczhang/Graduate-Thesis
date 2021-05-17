# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 22:42:54 2021

@author: Lenovo
"""
#评估各个社区分割算法的性能
import networkx as nx

#算法校验
#参数：G, communities, weight='weight'
from networkx.algorithms.community.quality import modularity
#参数：G, partition
from networkx.algorithms.community.quality import coverage
#参数：G, partition
from networkx.algorithms.community.quality import performance


#networkx 算法区，其中label算法是较新算法
#参数：G
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
#参数：G
from networkx.algorithms.community.label_propagation import label_propagation_communities
#参数：G，k（集团个数）
from networkx.algorithms.community.asyn_fluid import asyn_fluidc

#karateclub算法区
#参数：暂无
from karateclub import EdMot
#参数：暂无（较慢算法
from karateclub import GEMSEC
#参数：暂无
from karateclub import SCD


import pandas as pd
import json
import sys
#karateclub算法
def karateclucb_algorithmn(modeClass,G):
    splitter = modeClass()
    splitter.fit(G)
    member=splitter.get_memberships()
    n_member=[[] for item in range(max(member.values())+1)]
    for i in member.keys():
        n_member[member[i]].append(i)
    modu_coe=modularity(G, n_member, 'weight')
    perf_coe=performance(G, n_member)
    cove_coe=coverage(G,n_member)
    return {"modularity":modu_coe, "performance":perf_coe, "coverage":cove_coe}

#networkx算法
def networkx_algorithmn(modeClass,G,k):
    result=None
    if k==None:
        result=modeClass(G)
    else:
        result=modeClass(G,k)
    result=list(map(lambda x:list(x),result))
    modu_coe=modularity(G, result, 'weight')
    perf_coe=performance(G, result)
    cove_coe=coverage(G, result)
    return {"modularity":modu_coe, "performance":perf_coe, "coverage":cove_coe}

def main(args):
    data=pd.read_csv("./weighted/"+args+".csv",low_memory=False)
    G=nx.from_pandas_edgelist(data,"org_code","dst_code",True,nx.Graph)
    del data
    output={}
    print("EdMot")
    eva=karateclucb_algorithmn(EdMot,G)
    output["EdMot"]=eva
    print("GEMSEC")
    eva=karateclucb_algorithmn(GEMSEC,G)
    output["GEMSEC"]=eva
    print("SCD")
    eva=karateclucb_algorithmn(SCD,G)
    output["SCD"]=eva
    print("GMC")
    eva=networkx_algorithmn(greedy_modularity_communities,G,None)
    output["GMC"]=eva
    print("LPC")
    eva=networkx_algorithmn(label_propagation_communities,G,None)
    output["LPC"]=eva
    print("AFC")
    eva=networkx_algorithmn(asyn_fluidc,G,5)
    output["AFC"]=eva
    
    with open("./"+args+".json","w") as f:
        json.dump(output,f)
        print("finished")
    
    

#输入网络的文件名，如201901
if __name__ == '__main__':
    main(str(sys.argv[1]))