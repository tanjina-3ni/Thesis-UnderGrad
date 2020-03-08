# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:10:55 2020

@author: Aspire
"""

import pandas as pd

data = pd.read_csv('F:/Dataset/Dataset Version 4/csv_result-cleveland (corr).csv')

X =  data.drop('num', axis=1) #Drop specified labels from rows or columns. axis=1, col
target = data.iloc[:,-1]

# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=1) 


# training the model on training set 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test) 

# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)



from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


mat=(confusion_matrix(y_test, y_pred))
#print mat
df_cm = pd.DataFrame(mat, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (5,3))
sn.set(font_scale=1.2)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 15})# font size
#print(classification_report(y_test, y_pred))

