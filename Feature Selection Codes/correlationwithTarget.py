# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:32:50 2020

@author: Aspire
"""

import pandas as pd

data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (corr).csv')
X = data.iloc[:,0:47]  #independent columns
#y = data.iloc[:,-1]    #target column 'num' 
#print y
#get correlations of each features in dataset
corrmat = data.corr(method="pearson")

corrwithtarget=[]
for i in range(len(corrmat.columns)-1):
        if abs(corrmat.iloc[i, -1])>0.2:
            colname = corrmat.columns[i]
            corrwithtarget.append(colname)
print corrwithtarget