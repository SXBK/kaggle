# coding:utf-8
'''
average price that analyze utils may used
'''
__author='lzl'
__version="0.1"
import sys
sys.path.append("..")

from utils.select import *
import pandas as pd
import numpy as np
from datetime import datetime

#all price use here is close price
price = 'close_price'

def ave_list(data, len):
    if data.size >= len:
        return data[:len].mean() < data[0]
    else:
        return False

def average_price(stock_id, duration, date):
    sqlstr = 'select {} from stock_day where stock_id={} and idate <= "{}" ORDER BY idate limit {}'.format(price, stock_id, date, duration)
    count, begin = sql.exceQuery(sqlstr)
    if count < duration:
        print('Error: Didnt have enough data!')
        return;
    total = 0
    for p in begin:
        total = total + p[0]
    return total/duration

def ave_days(stock_id, duration, date):
    return average_price(stock_id, dutation, date)

def ave_days_10(stock_id, date):
    duration = 10
    return average_price(stock_id, dutation, date)

def ave_days_20(stock_id, date):
    duration = 20
    return average_price(stock_id, dutation, date)

def ave_days_30(stock_id, date):
    duration = 30
    return average_price(stock_id, dutation, date)

def ave_days_60(stock_id, date):
    duration = 60
    return average_price(stock_id, dutation, date)

def ave_weeks(stock_id, duration, date):
    duration = duration * 5
    return average_price(stock_id, duration, date)

if __name__  == '__main__':
    print(average_price(2, 20, '2017-08-23'))
       
