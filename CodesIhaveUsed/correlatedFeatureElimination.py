# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:22:17 2020

@author: Aspire
"""
import pandas as pd

data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (2).csv')


correlated_features = set()
X = data.iloc[:,0:47]
correlation_matrix = data.corr()
print correlation_matrix

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print correlated_features


