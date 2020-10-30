# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 22:39:16 2020

@author: Aspire
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:22:43 2020

@author: Aspire
"""
import pandas as pd

data = pd.read_csv('F:/Dataset/Mode/cleveland V2mode(corr2).csv')
data = data.apply(pd.to_numeric)
data.dtypes
X =  data.drop('num', axis=1) #Drop specified labels from rows or columns. axis=1, col
target = data.iloc[:,-1]

data = data.apply(pd.to_numeric)
data.dtypes
#print(data.shape)
#print(data.dtypes)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.30, random_state = 5)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(30, input_dim=8, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=1)


model.summary()
loss,acc = model.evaluate(X_test, y_test, verbose=0)
print('Loss = ',loss*100, 'Accuracy = ',acc*100)