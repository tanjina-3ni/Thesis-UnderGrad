# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:35:08 2020

@author: Aspire
"""
import pandas as pd

def confusionmatrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
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
    a=mat[1][1]*1.0
    b=mat[0][0]*1.0
    sen = (mat[1][0]+mat[1][1])*1.0 
    pre = (mat[0][1]+mat[1][1])*1.0 
    print "Accuracy: {:.2f}".format((a+b)/(mat[0][0]+mat[1][0]+mat[0][1]+mat[1][1])*100)
    print "Sensitivity or Recall: {:.2f}".format(a/sen*100)
    print "Precision: {:.2f}".format(a/pre*100)
    
    


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


confusionmatrix(y_test, y_pred)

