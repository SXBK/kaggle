# coding:utf-8
'''
Create database
'''
__author='lzl'
__version="0.1"
import datetime
import urllib
import logging
import tushare as ts
import pandas as pd

from logger import *
from sqlHandler import *

sconfig = {"addr":"localhost", "user":"root", "pwd":"root", "dbname":"stock", "port":"3306", "us":"/tmp/mysql.sock"}
sql = mysqlHandler(sconfig)
def log(s):
    print(s)

def getDataFromFetch(vdic, fetch):
    return pd.DataFrame(list(fetch), columns=vdic)

def getAllField(tname):
    qstr = "show create table {}".format(tname)
    count ,fetch = sql.exceQuery(qstr)
    dick = fetch[0][1]
    vdic = []
    for line in dick.split('\n'):
        if line.strip()[0] == "`":
            vdic.append(line.strip().lstrip("`")[:line.strip().lstrip("`").index("`")])
    return vdic

def getDataFromTable(tname):
    vdic = getAllField(tname)
    qstr = "select * from {}".format(tname)
    count ,fetch = sql.exceQuery(qstr)
    return getDataFromFetch(vdic, fetch)

def getAllIndustry():
    vdic = ['industry','stock_num']
    qstr = "select industry, count(*) from stock group by industry order by count(*) desc"
    count ,fetch = sql.exceQuery(qstr)
    return getDataFromFetch(vdic, fetch)

def getStockByIndustry(industry):
    vdic = getAllField('stock')
    qstr = "select * from stock where industry = {}".format(industry)
    count ,fetch = sql.exceQuery(qstr)
    return getDataFromFetch(vdic, fetch)

def getStockByCode(code):
    vdic = getAllField('stock_day')
    qstr = "select sd.* from stock_day sd left join stock s on s.id = sd.stock_id  where s.code = {}".format(code)
    count ,fetch = sql.exceQuery(qstr)
    return getDataFromFetch(vdic, fetch)

if __name__ == '__main__':
    #print(getDataFromTable('idx'))
    #print(getAllIndustry())
    print(getStockByCode(600161))
