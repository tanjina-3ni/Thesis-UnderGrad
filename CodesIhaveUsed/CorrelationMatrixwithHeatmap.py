# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:09:58 2020

@author: Aspire
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (5).csv')
X = data.iloc[:,0:47]  #independent columns
#y = data.iloc[:,-1]    #target column 'num' 
#print y
#get correlations of each features in dataset
corrmat = data.corr(method="pearson")
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=False,cmap="RdYlGn")


#correlated_features1 = []
#correlated_features2 = []
for i in range(len(corrmat.columns)):
    for j in range(i):
        if abs(corrmat.iloc[i, j]) > 0.7:
            colname = corrmat.columns[i]
            rowname = corrmat.columns[j]
#            correlated_features1.append(colname)
#            correlated_features2.append(rowname)
#            print correlated_features1,correlated_features2
            print colname,rowname,corrmat.iloc[i, j]

