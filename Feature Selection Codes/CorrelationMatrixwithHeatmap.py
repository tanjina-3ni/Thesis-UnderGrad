# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:09:58 2020

@author: Aspire
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (5).csv')
X = data.iloc[:,0:47]  #independent columns
y = data.iloc[:,-1]    #target column 
#get correlations of each features in dataset
corrmat = data.corr(method="pearson")
top_corr_features = corrmat.index
plt.figure(figsize=(40,40))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")