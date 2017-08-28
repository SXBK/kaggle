#coding:utf-8

import sys
sys.path.append("..")

from utils.select import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import date2num
import datetime

class bbdraw:
    def __init__(self, k):
        self.k=k
    def single(self, si):
        fig, ax = plt.subplots()
        ax.plot(si.idate, si.close_price, linewidth=2, label='CP')
        m = []
        for i in self.k:
            m = si.close_price.rolling(window=i, center=False).mean()
            ax.plot(si.idate, m, linewidth=2, label='M{}'.format(i))
        ax.legend(loc='upper right')
        plt.grid()
        plt.show()

    def double_date(self, si, sb, ni='A', nb='B'):
        fig, ax = plt.subplots()
        ax.plot(si.idate, si.close_price, linewidth=2, label='{}CP'.format(ni))
        ax.plot(sb.idate, sb.close_price, linewidth=2, label='{}CP'.format(nb))
        m = []
        for i in self.k:
            m = si.close_price.rolling(window=i, center=False).mean()
            ax.plot(si.idate, m, linewidth=1.5, label='{}M{}'.format(ni,i))
            m = sb.close_price.rolling(window=i, center=False).mean()
            ax.plot(sb.idate, m, linewidth=1.5, label='{}M{}'.format(nb,i))
        ax.legend(loc='upper right')
        plt.show()

    def double(self, si, sb, ni='A', nb='B'):
        fig, ax = plt.subplots()
        ax.plot(si.close_price, linewidth=2, label='{}CP'.format(ni))
        ax.plot(sb.close_price, linewidth=2, label='{}CP'.format(nb))
        m = []
        for i in self.k:
            m = si.close_price.rolling(window=i, center=False).mean()
            ax.plot(m, linewidth=1.5, label='{}M{}'.format(ni,i))
            m = sb.close_price.rolling(window=i, center=False).mean()
            ax.plot(m, linewidth=1.5, label='{}M{}'.format(nb,i))
        ax.legend(loc='upper left')
        plt.show()

if __name__=='__main__':
    df = getDataFromTable('stock_day')
    a = df[df.stock_id==364]
    print(a.dtypes)
    b = df[df.stock_id==3]
    bb = bbdraw([60, 180])
    bb.single(a)
    #bb.double_date(a, b, 'XX', 'ZZ')
