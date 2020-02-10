# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:43:14 2020

@author: Aspire
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (corr).csv')
X = data.drop('num', axis=1) #Drop specified labels from rows or columns. axis=1, col
target = data['num']
lr = LogisticRegression()
rfecvlr = RFECV(estimator=lr, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecvlr.fit(X, target)
#print rfecvlr.grid_scores_



rfc = RandomForestClassifier()
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, target)



plt.figure(figsize=(10, 5))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecvlr.grid_scores_) + 1), rfecvlr.grid_scores_, linewidth=3)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.show()
