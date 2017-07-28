'''data toolbox inlucding:
   -- loader
   -- writer
   -- downloader
   -- eraser
   etc
   market marks:
   -- sh, shanghai stock exchange, which is default
   -- sz, shenzhen stock exchange
'''
import os, sys
import numpy as np
import tushare as ts
import pandas as pd
import tables as tb

class stockdata(object):
    '''data class
    '''

    def __init__(self, *args):
        if len(args):
            self.TOPDIR = args[1]
        else:
            self.TOPDIR = "../data"
        if len(args) > 1:
            self.market = args[2]
        else:
            self.market = 'sh'
        if len(args) > 2:
            self.stock = args[3]
        else:
            self.stock = "000001" if self.market == 'sh' else "399001"

        self._setpath()

    def _isstockvalid(self):
        if isinstance(self.stock, str):
            if len(self.stock) == 6:
                raise(AttributeError("stock code error"))
        elif isinstance(self.stock, list):
            for _, s in self.stock.iteritems():
                if len(self.stock) != 6:
                    raise(AttributeError("stock code error"))
        else:
            raise(TypeError("No such stock initialization method"))


    def _setpath(self):
        '''set related path
        '''
        _isstockvalid()
        self.marketpath = os.path.join(self.TOPDIR, self.market)
        if isinstance(self.stock, str):
            self.stockpath = os.path.join(self.marketpath, self.stock)
        elif isinstance(self.stock, list):
            self.stockpath = [os.path.join(self.marketpath, s)
                              for s in self.stock]
        else:
            raise(TypeError("No such stock initialization method"))

    def setmarket(self, m):
        '''set market
        '''
        if m not in ['sh', 'sz']:
            raise(StandardError("No such market"))
        self.market = m
        self._setpath()

    def setstock(self, s):
        '''set stock
        '''
        self.stock = s
        self._setpath()
