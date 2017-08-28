# coding:utf-8
'''
7. industry

'''
import sys
sys.path.append("..")

from utils.select import *
import pandas as pd
import numpy as np
import ave
import fifo
import visible

class industry:
    def __init__(self, duration=None):
        self.sd = pd.read_csv('stock_day.csv')
        self.ids =  self.sd.stock_id.unique()
        self.inum = 110
        self.chin = None
        self.idate = None
        self.duration = duration
        self.st = None

    def get_all_industry(self):
        return getAllIndustry().industry.values

    def chose_industry(self):
        industry = getAllIndustry().industry.values
        df = self.sd
        sm = []
        mina = 99999
        st = getDataFromTable('stock')
        for i in range(len(industry)):
            ind = industry[i]
            #print(ind)
            ids = st[st.industry==industry[i]].id.values
            #print(ids)
            minl = 99999
            ss = [0]
            for id in ids:
                sc = df[df.stock_id==id].close_price.values
                #print(sc)
                if len(sc)< 1000:
                    continue
                if len(sc) < minl:
                    minl = len(sc)
                #print(minl, len(sc))
                ss = ss[-minl:] + sc[-minl:]
            ss = ss/len(ids)
            sm.append(ss)
            #print(mina, minl)
            if minl < mina:
                mina = minl
        print(mina)
        self.idate = self.sd.idate.unique()
        self.idate.sort()
        self.idate = self.idate[-mina:]
        self.idate = pd.to_datetime(self.idate)
        for i in range(len(sm)):
            #print(sm[i])
            sm[i]=sm[i][-mina:]
        sm = pd.DataFrame(data=sm, index=industry)
        return sm

    def chose_industry_duration(self, duration):
        industry = getAllIndustry().industry.values
        df = self.sd
        sm = []
        mina = duration
        st = getDataFromTable('stock')
        for i in range(len(industry)):
            ind = industry[i]
            #print(ind)
            ids = st[st.industry==industry[i]].id.values
            #print(ids)
            ss = [0]
            for id in ids:
                sc = df[df.stock_id==id].close_price.values
                #print(sc)
                if len(sc)< duration:
                    continue
                ss = ss[-duration:] + sc[-duration:]
            ss = ss/len(ids)
            sm.append(ss)
            #print(mina, minl)
        self.idate = self.sd.idate.unique()
        self.idate.sort()
        self.idate = self.idate[-duration:]
        self.idate = pd.to_datetime(self.idate)
        for i in range(len(sm)):
            #print(sm[i])
            sm[i]=sm[i][-duration:]
        sm = pd.DataFrame(data=sm, index=industry)
        return sm

    def chose_industry_corr(self):
        if self.chin is None:
            if self.duration is None:
                self.chin = self.chose_industry()
            else:
                self.chin = self.chose_industry_duration(self.duration)
        ind_corr = self.chin.T.corr()
        industry = self.get_all_industry()
        ff_max = fifo.fifo(10, 0)
        ff_min = fifo.fifo(10, 1)
        for i in range(110):
            for j in range(110):
                if i >= j :
                    continue
                if ind_corr[industry[i]][industry[j]] > ff_max.min().k:
                    ff_max.insert(ind_corr[industry[i]][industry[j]], [industry[i], industry[j]])
                if ind_corr[industry[i]][industry[j]] < ff_min.max().k:
                    ff_min.insert(ind_corr[industry[i]][industry[j]], [industry[i], industry[j]])
        print(ff_max.ff)
        print(ff_min.ff)

    def compair_industry(self, aa, bb):
        if self.chin is None:
            if self.duration is None:
                self.chin = self.chose_industry()
            else:
                self.chin = self.chose_industry_duration(self.duration)
        ind = self.chin.T
        a = pd.DataFrame(data=ind[unicode(aa,'utf8')].values, columns=['close_price'], index=ind.index)
        a['idate']=self.idate
        a.idate = a.idate.astype('object')
        b = pd.DataFrame(data=ind[unicode(bb,'utf8')].values, columns=['close_price'], index=ind.index)
        b['idate']=self.idate
        b.idate = b.idate.astype('object')
        bbd = visible.bbdraw([60])
        bbd.double_date(a,b,aa,bb)
        
    def detail(self, indus, duration):
        if self.st is None:
            self.st = getDataFromTable('stock')
        ids = self.st[self.st.industry==indus].id.values
        

if __name__=='__main__':
    sk = industry()

    #print(len(sk.chose_ave()))
    #sk.chose_industry_corr()
    sk.compair_industry('煤炭开采','化工机械')
