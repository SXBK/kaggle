# coding:utf-8
'''
1. std
2. e-b/b
3. guli/p
4. pe
5. v*diff
6. ave
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

class sketch:
    def __init__(self):
        self.sd = pd.read_csv('stock_day.csv')
        self.ids =  self.sd.stock_id.unique()
        self.inum = 110
        

#almost 1/2 stand up 60day mean
#almost 2/3 stand up 30day mean
    def chose_ave(self):
        duration = 60
        df = self.sd
        ss = []
        for id in self.ids:
            ss.append(df[df.stock_id==id].close_price.values)
        ok = []
        for i in range(len(ss)):
            if ave.ave_list(ss[i], duration):
                ok.append(self.ids[i])
        return ok

    def get_all_industry(self):
        return getAllIndustry().industry.values

    def chose_industry(self):
        industry = getAllIndustry().industry.values
        df = self.sd
        sm = []
        mina = 99999
        for i in range(len(industry)):
            ind = industry[i]
            #print(ind)
            st = getDataFromTable('stock')
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
                ss = ss[:minl] + sc[:minl]
            ss = ss/len(ids)
            sm.append(ss)
            #print(mina, minl)
            if minl < mina:
                mina = minl
        print(mina)
        #print(sm)
        for i in range(len(sm)):
            #print(sm[i])
            sm[i]=sm[i][:mina]
        sm = pd.DataFrame(data=sm, index=industry)
        return sm

    def chose_industry_corr(self):
        ind_corr = self.chose_industry().T.corr()
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
        ind = self.chose_industry().T
        a = pd.DataFrame(data=ind[unicode(aa,'utf8')].values, columns=['close_price'], index=ind.index)
        print(a.head())
        b = pd.DataFrame(data=ind[unicode(bb,'utf8')].values, columns=['close_price'], index=ind.index)
        bbd = visible.bbdraw([60])
        bbd.double(a,b,aa,bb)
        

if __name__=='__main__':
    sk = sketch()
    #print(len(sk.chose_ave()))
    #sk.chose_industry_corr()
    sk.compair_industry('煤炭开采','化工机械')
