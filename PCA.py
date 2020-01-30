# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:26:54 2020

@author: Aspire
"""

import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
url = 'F:/Dataset/Dataset Version 4/csv_result-cleveland.csv'
names = ['id','age','sex','cp','trestbps','htn','chol','cigs','years','fbs','famhist','restecg','ekgmo','ekgday','ekgyr','dig','prop','nitr','pro','diuretic','thaldur','thaltime','met','thalach','thalrest','tpeakbps','tpeakbpd','trestbpd','exang','xhypo','oldpeak','slope','ca','thal','cmo','cday','cyr','num','lmt','ladprox','laddist','cxmain','om1','rcaprox','rcadist','lvx3','lvx4','lvf']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:47]
Y = array[:,47]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)