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

#tables need update
#tnu = ['idx_day']
tnu = ['idx_day', 'stock_day']

def log(s):
    print(s)

def getUpdateContent(tname):
    qstr = "select idate from stock_day order by idate desc limit 1"
    count, fetch = sql.exceQuery(qstr)
    log(fetch[0][0])
    lastDay = fetch[0][0] + datetime.timedelta(days=1)
    log(lastDay.strftime("%Y-%m-%d"))
    switcher = {
        'idx_day': updateIdxDay,
        'stock_day': updateStockDay,
        }
    switcher.get(tname)(lastDay.strftime("%Y-%m-%d"))

def updateIdxDay(last):
    log('update idx day at {}'.format(last))
    qs = "select * from idx"
    index = sql.exceQuery(qs)
    for idx in index[1]:
        tf = ts.get_k_data(idx[1], start=last)
        for i, row in tf.iterrows():
            insert = "insert into idx_day (idx_id, idate, open_price, close_price, high, low, volume) values ({}, '{}', {}, {}, {}, {}, {});".format(idx[0], row.date, row.open, row.close, row.high, row.low, row.volume)
            log(insert)
            count, fetch = sql.exceQuery(insert)
            if count != 1:
                sql.closeConn()
                return -1
    sql.commitConn()
    return 1

def updateStockDay(last):
    log('update stock day')
    qs = "select id, code from stock"
    index = sql.exceQuery(qs)
    for idx in index[1]:
        log(idx)
        tf = ts.get_k_data(idx[1], start=last)
        for i, row in tf.iterrows():
            insert = "insert into stock_day (stock_id, idate, open_price, close_price, high, low, volume) values ({}, '{}', {}, {}, {}, {}, {});".format(idx[0], row.date, row.open, row.close, row.high, row.low, row.volume)
            log(insert)
            count, fetch = sql.exceQuery(insert)
            if count != 1:
                sql.closeConn()
                return -1
    sql.commitConn()
    return 1

def updateAll():
    for tname in tnu:
        upContent= getUpdateContent(tname)
        if upContent != 1:
            log('update error at update {}'.format(tname))
    sql.closeConn()

if __name__=='__main__':
    updateAll()
