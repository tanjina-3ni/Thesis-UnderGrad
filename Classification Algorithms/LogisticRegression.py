# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:35:08 2020

@author: Aspire
"""
import pandas as pd



data = pd.read_csv('F:/Dataset/Mode/cleveland V2mode(corr2).csv')

X =  data.drop('num', axis=1) #Drop specified labels from rows or columns. axis=1, col
target = data.iloc[:,-1]

# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=1) 
logreg = LogisticRegression()
logreg.fit(X_train, y_train) 

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

