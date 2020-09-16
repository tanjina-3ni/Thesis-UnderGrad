# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:10:55 2020

@author: Aspire
"""

import pandas as pd

# %%
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
    print "Accuracy: ",(a+b)/(mat[0][0]+mat[1][0]+mat[0][1]+mat[1][1])*100
    print "Sensitivity or Recall: ",a/sen*100
    print "Precision: ",a/pre*100
    
    
# %%   
    

data = pd.read_csv('F:/Dataset/Mode/cleveland V2mode(corr2).csv')

X =  data.drop('num', axis=1) #Drop specified labels from rows or columns. axis=1, col
target = data.iloc[:,-1]

# %%

# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=1) 

#%%

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='rbf', C=1, gamma=0.1, probability=True).fit(X, target)
clf.fit(X_train, y_train) 

# making predictions on the testing set 
y_pred = clf.predict(X_test) 

# %%

# comparing actual response values (y_test) with predicted response values (y_pred) 
confusionmatrix(y_test, y_pred)


# %%
from sklearn.model_selection import cross_val_score
#k_fold = KFold(len(target), n_folds=5, shuffle=True, random_state=1)

cv = cross_val_score(clf, X, target, cv=5)

s = 0
for i in range(0,len(cv)):
    s = s + cv[i]
print "Accuracy after cross validation : ", (s/len(cv))*100


