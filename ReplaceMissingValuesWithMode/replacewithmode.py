# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:51:52 2020

@author: Aspire
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

df = pd.read_csv('F:/Dataset/Mode/cleveland V2.csv')

#features = df.iloc[:,:-1].to_numpy()
##print features
#dependent=df.iloc[:,-1:].to_numpy()
##print dependent[1,0]
#imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
#imputer.fit(features[7,:])
#features[7,:]=imputer.fit_transform(features[7,:])
#
##print features
#df.to_csv('mode.csv')

mean=df["years"].mean()
print mean
rp=df["years"].replace(np.NaN,mean)
print rp
#df.to_csv('mode.csv')
#mode=df[""].mode()
#print "cigs = ",mode
#rp=df["cigs"].replace(np.NaN,mode)
#print rp
#print "\n"

#years=df["years"].mode()
#print "years = ",years
#print "\n"
#
#ca=df["ca"].mode()
#print "ca = ",ca
#print "\n"
#
#thal=df["thal"].mode()
#print "thal",thal
#print "\n"
#
#dig=df["dig"].mode()
#print "dig",dig
#print "\n"
#
#nitr=df["nitr"].mode()
#print "nitr",nitr
#print "\n"
#
#pro=df["pro"].mode()
#print "pro",pro
#print "\n"
#
#diuretic=df["diuretic"].mode()
#print "diuretic",diuretic
