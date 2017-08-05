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

def getAllField(tname):
    qstr = "show create table {}".format(tname)
    count ,fetch = sql.exceQuery(qstr)
    dick = fetch[0][1]
    vdic = []
    for line in dick.split('\n'):
        if line.strip()[0] == "`":
            vdic.append(line.strip().lstrip("`")[:line.index("`")])
    return vdic

def getDataFromMysql(tname):
    vdic = getAllField(tname)
    qstr = "select * from {}".format(tname)
    count ,fetch = sql.exceQuery(qstr)
    data = pd.DataFrame(list(fetch), columns=vdic)
    return data

if __name__ == '__main__':
    print(getDataFromMysql('idx'))
