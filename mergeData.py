#程序以日为粒度获取每日航空网络数据
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

#输入日期
date=[list(range(20190101,20190132))]
date+=[list(range(20190201,20190229))]
date+=[list(range(20190301,20190332))]
date+=[list(range(20190401,20190431))]
date+=[list(range(20190501,20190532))]
date+=[list(range(20190601,20190631))]
date+=[list(range(20190701,20190732))]
date+=[list(range(20190801,20190832))]
date+=[list(range(20190901,20190931))]
date+=[list(range(20191001,20191032))]
date+=[list(range(20191101,20191131))]
date+=[list(range(20191201,20191232))]

date+=[list(range(20200101,20200132))]
date+=[list(range(20200201,20200230))]
date+=[list(range(20200301,20200332))]
date+=[list(range(20200401,20200431))]
date+=[list(range(20200501,20200532))]
date+=[list(range(20200601,20200631))]
date+=[list(range(20200701,20200732))]
date+=[list(range(20200801,20200832))]
date+=[list(range(20200901,20200931))]
date+=[list(range(20201001,20201032))]
date+=[list(range(20201101,20201131))]
date+=[list(range(20201201,20201232))]
#输入原始数据
path=['flightlist_20190101_20190131','flightlist_20190201_20190228','flightlist_20190301_20190331',
      'flightlist_20190401_20190430','flightlist_20190501_20190531','flightlist_20190601_20190630',
      'flightlist_20190701_20190731','flightlist_20190801_20190831','flightlist_20190901_20190930',
      'flightlist_20191001_20191031','flightlist_20191101_20191130','flightlist_20191201_20191231',
      'flightlist_20200101_20200131','flightlist_20200201_20200229','flightlist_20200301_20200331',
      'flightlist_20200401_20200430','flightlist_20200501_20200531','flightlist_20200601_20200630',
      'flightlist_20200701_20200731','flightlist_20200801_20200831','flightlist_20200901_20200930',
      'flightlist_20201001_20201031','flightlist_20201101_20201130','flightlist_20201201_20201231']
print("begin")
#处理
for index,item in enumerate(path):
    data=pd.read_csv(item+'.csv')
    data['day']=data['day'].apply(lambda x : (x.split(' ')[0]).replace('-',''))
    for day in date[index]:
        temp=data[data['day']==str(day)]
        lines=[]
        for index,row in temp.iterrows():
            row=dict(row)
            if row["origin"] not in dict_data.keys() or row["destination"] not in dict_data.keys():
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
        lines.to_csv("./handle/"+str(day)+".csv")
        print(day)
print("finished")