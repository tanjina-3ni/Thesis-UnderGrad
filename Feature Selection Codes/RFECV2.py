# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:43:14 2020

@author: Aspire
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (4).csv')

X = data.drop('num', axis=1) #Drop specified labels from rows or columns. axis=1, col
target = data['num']


lr = LogisticRegression()
rfecvlr = RFECV(estimator=lr, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecvlr.fit(X, target)
#print rfecvlr.grid_scores_

svc = SVC(kernel="linear")
rfecvsvm = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecvsvm.fit(X, target)


rfc = RandomForestClassifier(n_estimators=60)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, target)


plt.figure(figsize=(10, 5))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecvlr.grid_scores_) + 1), rfecvlr.grid_scores_,  color='RED', linewidth=3)
plt.plot(range(1, len(rfecvsvm.grid_scores_) + 1), rfecvsvm.grid_scores_,  color='ORANGE', linewidth=3)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='BLUE', linewidth=3)

plt.show()
