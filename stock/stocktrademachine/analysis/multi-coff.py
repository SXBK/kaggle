import sys
sys.path.append("..")

from utils.select import *
import pandas as pd
import numpy as np
import thread

split = 8

sd = getDataFromTable('stock_day')

date = np.sort(sd.idate.unique())[-300:][::-1]
'''
count, fetch = sql.exceQuery('select stock_id from stock_day where idate > "2016-05-15" group by stock_id having count(*) >= 300')
sids = getDataFromFetch(['id'], fetch)
sids.to_csv('sids_300.csv')
'''
sids = pd.read_csv('sids_300.csv')


n = 0

def proc(ids):
    for i in ids:
        for d in date:
            a = sd.loc[(sd['stock_id'] == i) & (sd['idate'] == d)]
            print(i, d, a.shape)
            if a.shape[0] == 1:
                oo.loc[i][d] = a.close_price.values[0]
    n = n + 1
    
oo = pd.DataFrame(index=sids.id.values, columns=date)

sid = np.split(sids.id, split)

for ids in sid:
    thread.start_new_thread( proc, (ids,) )

while n < split:
    pass

print(oo.head())

oo.to_csv('300.csv')
        
corr = oo.corr()

cov = oo.cov()


