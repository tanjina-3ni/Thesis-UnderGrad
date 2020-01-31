# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:29:47 2020

@author: Aspire
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt

data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (2).csv')
X = data.iloc[:,0:47]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_classif, k=14)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
  #print 10 best features

l=[]
l=featureScores.nlargest(14,'Score')
print(l)
#featureScores.nlargest(14,'Score')
ax = plt.gca()
l.plot(kind='bar',x='Specs',y='Score',ax=ax)
#
#plt.show()