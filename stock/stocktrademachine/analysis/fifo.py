# coding:utf-8
import pandas as pd
import numpy as np

class fifo():
    def __init__(self, size, mix):
        self.size = size
        self.mix = mix
        if mix == 0:
            self.ff = pd.DataFrame(data=np.zeros(shape=(size,2)),index=range(size),columns=['k','v'] ,dtype=object)
        else:
            self.ff = pd.DataFrame(data=np.ones(shape=(size,2)),index=range(size),columns=['k','v'] ,dtype=object)*9999

    def max(self):
        return self.ff.loc[0]

    def min(self):
        return self.ff.loc[self.size-1]

    def insert(self, k, v):
        if self.mix == 0:
            return self.insert_max(k, v)
        else:
            return self.insert_min(k, v)
    
    def insert_max(self, k, v):
        now = 0
        for i in range(self.size):
            now = i
            if self.ff.loc[i].k < k:
                break
        for i in range(self.size-1, now, -1):
            self.ff.loc[i]=self.ff.loc[i-1]
        self.ff.loc[now].k=k
        self.ff.loc[now].v=v

    def insert_min(self, k, v):
        now = 0
        for i in range(self.size):
            now = i
            if self.ff.loc[i].k < k:
                break
            if i == self.size-1:
                now = now + 1
        if now > 0:
            now = now - 1
            for i in range(0, now, 1):
                self.ff.loc[i]=self.ff.loc[i+1]
            self.ff.loc[now].k=k
            self.ff.loc[now].v=v

    def ffprint(self):
        pass


if __name__=='__main__':
    ff = fifo(3,1)
    ff.insert(1,2)
    print(ff.ff)
    ff.insert(3,[3, 5])
    print(ff.ff)
    ff.insert(4,[3, 5])
    print(ff.ff)
    ff.insert(2,[5])
    print(ff.max().k)

