#融合Gdelt和航空网络数据
import pandas as pd
header_names=['GlobalEventID', 'Day', 'MonthYear', 'Year', 'FractionDate',
       'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
       'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
       'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code', 'Actor2Code',
       'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
       'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
       'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code', 'IsRootEvent',
       'EventCode', 'EventBaseCode', 'EventRootCode', 
       'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
       'NumArticles', 'AvgTone', 'Actor1Geo_Type', 'Actor1Geo_Fullname',
       'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 
       'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID', 'Actor2Geo_Type',
       'Actor2Geo_Fullname', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code',
        'Actor2Geo_Lat', 'Actor2Geo_Long',
       'Actor2Geo_FeatureID', 'ActionGeo_Type', 'ActionGeo_Fullname',
       'ActionGeo_CountryCode', 'ActionGeo_ADM1Code',
       'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID', 'DATEADDED',
       'SOURCEURL.']
date=list(range(20200101,20200132))
date+=list(range(20200201,20200230))
date+=list(range(20200301,20200332))
date+=list(range(20200401,20200431))

output=[]
tag_dict={}
for idt,day in enumerate(date):
    data=pd.read_table("./events/"+str(day)+".export.CSV",engine='python',header=None, names=header_names)
    #筛选需要的数据列并清空空缺数据列
    data=data[["Actor1CountryCode","Actor2CountryCode",'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources','NumArticles', 'AvgTone']]
    data=data.dropna(axis=0,how='any')
    #读取国家编号文件，进行二字码编码
    country=pd.read_csv("country.csv")
    country=country[country["Continent_Code"]=="EU"]
    #中间文件：先对于actor1进行连接并命名
    temp_data=pd.merge(data,country[["Continent_Code","Three_Letter_Country_Code","Two_Letter_Country_Code"]],
                  left_on="Actor1CountryCode",right_on="Three_Letter_Country_Code",how="inner")
    del temp_data["Three_Letter_Country_Code"]
    del temp_data["Continent_Code"]
    temp_data=temp_data.rename(columns={"Two_Letter_Country_Code":"Actor1Country2Code"})
    #最后数据：对于actor2进行连接并命名
    n_data=pd.merge(temp_data,country[["Continent_Code","Three_Letter_Country_Code","Two_Letter_Country_Code"]],
                  left_on="Actor2CountryCode",right_on="Three_Letter_Country_Code",how="inner")
    del n_data["Three_Letter_Country_Code"]
    del n_data["Continent_Code"]
    n_data=n_data.rename(columns={"Two_Letter_Country_Code":"Actor2Country2Code"})
    #准备压缩数据列：边整合
    further_group=[]
    
    for index, row in n_data.iterrows():
        row=dict(row)
        a1=row["Actor1Country2Code"]
        a2=row["Actor2Country2Code"]
        if a1+"-"+a2 in tag_dict.keys():
            row["index"]=a1+"-"+a2
        elif a2+"-"+a1 in tag_dict.keys():
            row["index"]=a2+"-"+a1
        else:
            tag_dict[a1+"-"+a2]=1
            row["index"]=a1+"-"+a2
        further_group.append(row)
    further_group=pd.DataFrame(further_group)
    temp=further_group.groupby(["index"])[["QuadClass","GoldsteinScale","NumMentions","NumSources","NumArticles","AvgTone"]].mean()
    flight=pd.read_csv("../FlightData/day/"+str(day)+".csv")
    flight=flight[(flight["org_continent"]=="EU")&(flight["dst_continent"]=="EU")]
    flight_dict={}
    for index,row in flight.iterrows():
        row=dict(row)
        a1=row["org_country"]
        a2=row["dst_country"]
        if a1+"-"+a2 in flight_dict.keys():
            flight_dict[a1+"-"+a2]=flight_dict[a1+"-"+a2]+1
        elif a2+"-"+a1 in flight_dict.keys():
            flight_dict[a2+"-"+a1]=flight_dict[a2+"-"+a1]+1
        else:
            flight_dict[a1+"-"+a2]=1
    
    lines=[]
    for index,row in temp.iterrows():
        row=dict(row)
        if index in flight_dict.keys():
            row["lines"]=flight_dict[index]
        else:
            sp=index.split("-")
            new_index=sp[1]+"-"+sp[0]
            if new_index in flight_dict.keys():
                row["lines"]=flight_dict[new_index]
            else:
                continue
        row["date"]=idt
        row["flow"]=index
        lines.append(row)
    output+=lines
    print(day)
pd.DataFrame(output).to_csv("output.csv")