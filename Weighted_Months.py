#计算带权网络
import pandas as pd
import networkx as nx
date=list(range(201901,201913))
date+=list(range(202001,202013))

for day in date:
    print(day,"begin")
    data=pd.read_csv("./month/"+str(day)+".csv",low_memory=False)
    data=data[(data["org_continent"]=="EU")&(data["dst_continent"]=="EU")]
    G=nx.from_pandas_edgelist(data,"origin","destination",True,nx.Graph)
    G=max(nx.connected_components(G),key=len)
    lines=[]
    for index, row in data.iterrows():
        row=dict(row)
        if row["origin"] not in G or row["destination"] not in G:
            continue
        lines.append(row)
    data1=pd.DataFrame(lines)
    print("network calculation")
    airdict={}
    hash_edges={}
    for index, row in data1.iterrows():
        row=dict(row)
        if row["origin"] not in airdict.keys():
            num=len(airdict)
            airdict[row["origin"]]=num

        if row["destination"] not in airdict.keys():
            num=len(airdict)
            airdict[row["destination"]]=num

        row["org_code"]=airdict[row["origin"]]
        row["dst_code"]=airdict[row["destination"]]

        if str(row["org_code"])+"-"+str(row["dst_code"]) in hash_edges.keys():
            hash_edges[str(row["org_code"])+"-"+str(row["dst_code"])]["weight"]+=1
        elif str(row["dst_code"])+"-"+str(row["org_code"]) in hash_edges.keys():
            hash_edges[str(row["dst_code"])+"-"+str(row["org_code"])]["weight"]+=1
        else:
            hash_edges[str(row["org_code"])+"-"+str(row["dst_code"])]=row
            hash_edges[str(row["org_code"])+"-"+str(row["dst_code"])]["weight"]=1

    output=[]
    print("reorganization")
    for item in hash_edges.keys():
        output.append(hash_edges[item])
    output=pd.DataFrame(output)
    output.to_csv("./weighted/"+str(day)+".csv")
    print(day,"end")
