# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:36:22 2020

@author: Aspire
"""

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (2).csv')
X = data.iloc[:,0:47]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 14)
fit = rfe.fit(X, y)
dfscores = pd.DataFrame(fit.ranking_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Ranking']  #naming the dataframe columns
print(featureScores.nsmallest(14,'Ranking'))