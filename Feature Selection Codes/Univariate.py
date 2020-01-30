# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:58:31 2020

@author: Aspire
"""

# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# load data
filename = 'F:/Dataset/Dataset Version 4/csv_result-cleveland.csv'
names = ['id','age','sex','cp','trestbps','htn','chol','cigs','years','fbs','famhist','restecg','ekgmo','ekgday','ekgyr','dig','prop','nitr','pro','diuretic','thaldur','thaltime','met','thalach','thalrest','tpeakbps','tpeakbpd','trestbpd','exang','xhypo','oldpeak','slope','ca','thal','cmo','cday','cyr','num','lmt','ladprox','laddist','cxmain','om1','rcaprox','rcadist','lvx3','lvx4','lvf']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:47]
Y = array[:,47]
# feature extraction
test = SelectKBest(score_func=f_classif, k=14)
fit = test.fit(X, Y)

# summarize scores
set_printoptions(precision=2)
print(fit.scores_)
#features = fit.transform(X)
# summarize selected features
#print(features[0:15,:])