# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:33:40 2016
百度搜索结果爬虫
@author: yaohongfu
"""
import urllib.request 
from bs4 import BeautifulSoup 
#import pymysql
#import zlib
import gzip#解压缩的包
import re
#解压函数
def ungzip(data):
    try:        # 尝试解压
        #print('正在解压.....')
        data = gzip.decompress(data)
        #print('解压完毕!')
    except:
        pass
        #print('未经压缩, 无需解压')
    return data
#转url中文码
#urllib.parse.unquote("%D4%CB%B6%AF%D0%AC%C5%AE",encoding='gbk')
#urllib.parse.unquote("%E5%A4%A9%E6%89%8D",encoding="utf-8")
def bildurl(key):
    p={"wd":key}
    keyword=urllib.parse.urlencode(p)
    search_url='http://www.baidu.com/s?key&ct=1&rn=50'
    s=search_url.replace('key',keyword)
    #print(s)
    return s

#返回text中满足规则的子串的列表 
def getList(regex,text):
    arr=[]
    res=re.findall(regex,text)
    if res:
        for r in res:
            #print("r是",r)
            arr.append(r)
    return arr

#函数化
url=[]
def search(key):
    global url
    j=1
    star_url=bildurl(key)
    use_url=star_url
    #star_url="http://www.baidu.com/s?wd=%E9%A9%AC%E6%9C%AF&pn=50&oq=%E9%A9%AC%E6%9C%AF&ct=1&rn=50&ie=utf-8&usm=3&rsv_pq=86733f3c00016391&rsv_t=9fca3WTQw38SjTmQEcLaJgaCvOZTAeg75p7gaDZ9Ce1y61SqgvgoEacg36c&rsv_page=1"
    url.append(use_url)
    for count in list(range(20)):
        req=urllib.request.Request(use_url,headers={"User-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:36.0) Gecko/20100101 Firefox/36.0","Referer":"http://xxx.yyy.com/test0","Accept":"textml,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language":"en-US,en;q=0.5","Accept-Encoding":"gzip, deflate","Connection":"keep-alive","Content-Type":"application/x-www-form-urlencoded",})
        response = urllib.request.urlopen(req,timeout=20)
        html=response.read()
        bytes_data=ungzip(html)
        soup=BeautifulSoup(bytes_data,"lxml")
        file=open("savebaidu.txt",'a',encoding='utf-8')
        catch_content=[]
        for con in soup.select('div[class="c-abstract"]'):
            catch_content.append(con.get_text())
        #j=1
        for i in catch_content:
            file.write(str(j)+"\t"+"page"+str(count+1)+"\t"+key+"\t"+i+'\n')
            j+=1
        file.close()
        if use_url==star_url:
            next_page='http://www.baidu.com'+soup('a',{'href':True,'class':'n'})[0]['href']
        else:
            try:
                next_page='http://www.baidu.com'+soup('a',{'href':True,'class':'n'})[1]['href']
            except:
                print("只有"+str(count+1)+"页")
                break
        use_url=next_page
        url.append(next_page)
    
url=[]
wordlist_ad_new=[r'四维图新',r'永辉超市']

for wd in wordlist_ad_new:
    search(wd)
    
    
    
#连接数据库
#import pymysql
#filepath=r"..\end.txt"
#df_data=pd.read_table(filepath,sep='\t',names =['num','page','keyword','content','classlabel'])
#df_xls=pd.read_excel(r'E:\spide_baidu\tiyu.xlsx')
#conn = pymysql.connect(host='192.168.62.173', port=3308,user='root',passwd='123456',db='test',charset='UTF8')
#df_data.to_sql('baidu_yuliao_1026',conn,flavor='mysql',if_exists='append',index=False,chunksize=100000)

#conn.close()