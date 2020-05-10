# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:14:57 2020

@author: Aspire
"""

import numpy as np
import pandas as pd

df = pd.read_csv('F:/Dataset/Mode/cleveland V2.csv')


dependent=df.iloc[:,-1:].to_numpy()
for i in range(0,282):
    if dependent[i,0]!=0:
        dependent[i,0]=1

#print dependent[:,0]
#print dependent
df.iloc[:,-1:]=dependent
df.to_csv('numedited.csv')