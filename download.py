#下载Gdelt数据
#http://data.gdeltproject.org/events/20200101.export.CSV.zip
import requests
import zipfile
date=list(range(20200101,20200132))
date+=list(range(20200201,20200230))
date+=list(range(20200301,20200332))
date+=list(range(20200401,20200431))

for index,day in enumerate(date): 
    url = 'http://data.gdeltproject.org/events/'+str(day)+'.export.CSV.zip' 
    r = requests.get(url) 
    with open("./gdelt/"+str(day)+".zip", "wb") as code:
        code.write(r.content)
    with zipfile.ZipFile("./gdelt/"+str(day)+".zip",'r') as f:
        f.extractall('events')
    if index%10==0:
        print(day)