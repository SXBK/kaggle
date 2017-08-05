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


def create_all_index():
    index = ts.get_index()
    index['title'] = index['name']
    for idx, row in index.iterrows():
        insert = "insert into idx (`code`, `name`) values ( '{}', '{}');".format(row.code, row.title)
        log(insert)
        count, fetch = sql.exceQuery(insert)
        log("excute count {}".format(count))
        if count != 1:
            sql.closeConn()
            return -1
    sql.commitConn()
    sql.closeConn()

    
def create_all_index_day():
    qs = "select * from idx"
    index = sql.exceQuery(qs)
    for idx in index[1]:
        log(idx[2])
        tf = ts.get_k_data(idx[1], start='2011-01-01')
        for i, row in tf.iterrows():
            insert = "insert into idx_day (idx_id, idate, open_price, close_price, high, low, volume) values ({}, '{}', {}, {}, {}, {}, {});".format(idx[0], row.date, row.open, row.close, row.high, row.low, row.volume)
            log(insert)
            count, fetch = sql.exceQuery(insert)
            if count != 1:
                sql.closeConn()
                return -1
    sql.commitConn()
    sql.closeConn()

def create_all_stock():
    bas = ts.get_stock_basics()
    bas['title'] = bas['name']
    bas['code'] = bas.index
    for idx, row in bas.iterrows():
        insert = "insert into stock (code, name, industry, area, fixedAssets, liquidAssets, outstanding, totals, totalAssets, reserved, reservedPerShare, esp, bvps, pb, timeToMarket, undp, perundp, rev, profit, gpr, npr, holders) values ('{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, '{}', {}, {}, {}, {}, {}, {}, {});".format(row.code, row.title, row.industry, row.area, row.fixedAssets, row.liquidAssets, row.outstanding, row.totals, row.totalAssets, row.reserved, row.reservedPerShare, row.esp, row.bvps, row.pb, row.timeToMarket, row.undp, row.perundp, row.rev, row.profit, row.gpr, row.npr, row.holders)
        log(insert)
        count, fetch = sql.exceQuery(insert)
        log("excute count {}".format(count))
        if count != 1:
            sql.closeConn()
            return -1
    sql.commitConn()
    sql.closeConn()

def create_all_stock_info():
    bas = ts.get_stock_basics()
    bas['title'] = bas['name']
    now = datetime.datetime.now().strftime("%Y%M%d")
    for idx, row in bas.iterrows():
        insert = "insert into stock (code, name, industry, area, fixedAssets, liquidAssets) values ('{}', '{}', '{}', '{}', {}, {});".format(row.code, row.title, row.industry, row.area, row.fixedAssets, row.liquidAssets)
        log(insert)
        count, fetch = sql.exceQuery(insert)
        log("excute count {}".format(count))
        if count != 1:
            sql.closeConn()
            return -1
    sql.commitConn()
    sql.closeConn()

def create_all_stock_day():
    qs = "select id, code from stock"
    index = sql.exceQuery(qs)
    for idx in index[1]:
        log(idx)
        tf = ts.get_k_data(idx[1], start='2011-01-01')
        for i, row in tf.iterrows():
            insert = "insert into stock_day (stock_id, idate, open_price, close_price, high, low, volume) values ({}, '{}', {}, {}, {}, {}, {});".format(idx[0], row.date, row.open, row.close, row.high, row.low, row.volume)
            log(insert)
            count, fetch = sql.exceQuery(insert)
            if count != 1:
                sql.closeConn()
                return -1
    sql.commitConn()
    sql.closeConn()

def create_all_stock():
    bas = ts.get_stock_basics()
    bas['title'] = bas['name']
    bas['code'] = bas.index
    for idx, row in bas.iterrows():
        insert = "insert into stock (code, name, industry, area, fixedAssets, liquidAssets, outstanding, totals, totalAssets, reserved, reservedPerShare, esp, bvps, pb, timeToMarket, undp, perundp, rev, profit, gpr, npr, holders) values ('{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, '{}', {}, {}, {}, {}, {}, {}, {});".format(row.code, row.title, row.industry, row.area, row.fixedAssets, row.liquidAssets, row.outstanding, row.totals, row.totalAssets, row.reserved, row.reservedPerShare, row.esp, row.bvps, row.pb, row.timeToMarket, row.undp, row.perundp, row.rev, row.profit, row.gpr, row.npr, row.holders)
        log(insert)
        count, fetch = sql.exceQuery(insert)
        log("excute count {}".format(count))
        if count != 1:
            sql.closeConn()
            return -1
    sql.commitConn()
    sql.closeConn()

if __name__=="__main__":
    #create_all_index()
    #create_all_index_day()
    #create_all_stock()
    create_all_stock_day()
