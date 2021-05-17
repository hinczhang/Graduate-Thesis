#以月为粒度处理数据
from traffic.data import airports
import pandas as pd
import json
import sys
import os
import math
dict_data={}
p = 'airports.json'
with open(p,'r', encoding='UTF-8') as f:
     dict_data = json.load(f)
country=pd.read_csv("country.csv")

path=['flightlist_20190101_20190131','flightlist_20190201_20190228','flightlist_20190301_20190331',
      'flightlist_20190401_20190430','flightlist_20190501_20190531','flightlist_20190601_20190630',
      'flightlist_20190701_20190731','flightlist_20190801_20190831','flightlist_20190901_20190930',
      'flightlist_20191001_20191031','flightlist_20191101_20191130','flightlist_20191201_20191231',
      'flightlist_20200101_20200131','flightlist_20200201_20200229','flightlist_20200301_20200331',
      'flightlist_20200401_20200430','flightlist_20200501_20200531','flightlist_20200601_20200630',
      'flightlist_20200701_20200731','flightlist_20200801_20200831','flightlist_20200901_20200930',
      'flightlist_20201001_20201031','flightlist_20201101_20201130','flightlist_20201201_20201231']
date=list(range(201901,201913))
date+=list(range(202001,202013))

print("begin")
i_index=0
for index,item in enumerate(path):
    data=pd.read_csv(item+'.csv')
    data['day']=data['day'].apply(lambda x : (x.split(' ')[0]).replace('-',''))
    lines=[]
    for index,row in data.iterrows():
        row=dict(row)
        if row["origin"] not in dict_data.keys() or row["destination"] not in dict_data.keys():
            print(row["origin"],row["destination"])
            continue
        org=dict_data[row["origin"]]
        dst=dict_data[row["destination"]]
        row["org_lat"]=org["lat"]
        row["org_lon"]=org["lon"]
        row["org_name"]=org["name"]
        row["org_country"]=org["country"]
        row["org_state"]=org["state"]
        row["org_city"]=org["city"]
        org_continent=list(country[country["Two_Letter_Country_Code"]==org["country"]]["Continent_Code"])[0]
            
        if org_continent=="NA_A":
            #print(org_continent)
            org_continent="NA"
        row["org_continent"]=org_continent
            
        row["dst_lat"]=dst["lat"]
        row["dst_lon"]=dst["lon"]
        row["dst_name"]=dst["name"]
        row["dst_country"]=dst["country"]
        row["dst_state"]=dst["state"]
        row["dst_city"]=dst["city"]
        #print(dst["country"])
        dst_continent=list(country[country["Two_Letter_Country_Code"]==dst["country"]]["Continent_Code"])[0]
        if dst_continent=="NA_A":
            #print(dst_continent)
            dst_continent="NA"
        row["dst_continent"]=dst_continent
        lines.append(row)
        
    lines=pd.DataFrame(lines)
    lines.to_csv("./month_handle/"+str(date[i_index])+".csv")
    print(date[i_index])
    i_index+=1
print("finished")