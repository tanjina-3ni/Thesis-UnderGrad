# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:12:49 2020

@author: Aspire
"""

from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = 'F:/Dataset/Dataset Version 4/csv_result-cleveland.csv'
names = ['id','age','sex','cp','trestbps','htn','chol','cigs','years','fbs','famhist','restecg','ekgmo','ekgday','ekgyr','dig','prop','nitr','pro','diuretic','thaldur','thaltime','met','thalach','thalrest','tpeakbps','tpeakbpd','trestbpd','exang','xhypo','oldpeak','slope','ca','thal','cmo','cday','cyr','num','lmt','ladprox','laddist','cxmain','om1','rcaprox','rcadist','lvx3','lvx4','lvf']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:47]
Y = array[:,47]
# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 14)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
l=[]
l=fit.ranking_
print (l)

for i in l:
    #print i
    if i==1:
        #print 'ok'
        print 