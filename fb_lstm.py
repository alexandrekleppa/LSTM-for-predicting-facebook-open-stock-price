# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:20:40 2020

@author: Alexandre Kleppa
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_train=pd.read_csv('MacroTrends_Data_Download_FB_train.csv')

open_stock_price_train = data_train['open'].values
open_stock_price_train=open_stock_price_train.reshape((open_stock_price_train.size),1)

#scaling 
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
sc.fit(open_stock_price_train)
open_stock_price_train=sc.transform(open_stock_price_train)



"""
The ideia is predicting two weeks (10 working days) of the opened facebook stock price based on the 
previous 90 days of its value
"""
X_train=[]
y_train=[]

total_days=np.size(open_stock_price_train)

for i in range (90, total_days-10):
    X_train.append(open_stock_price_train[i-90:i])
    y_train.append(open_stock_price_train[i:i+10])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_test_2w=open_stock_price_train[(total_days-100):(total_days-10)]
X_test_2w=X_test_2w.reshape(90,1)
X_test_2w=np.transpose(X_test_2w)

y_test_2w=open_stock_price_train[(total_days-10):total_days]
y_test_2w=y_test_2w.reshape(10,1)
y_test_2w=np.transpose(y_test_2w)

y_train=y_train.reshape((y_train.shape[0]),(y_train.shape[1]))
y_test_2w=y_test_2w.reshape((y_test_2w.shape[0]),(y_test_2w.shape[1]))

#building the LSTM

from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM

#initialize the RNN
regressor=Sequential()

#input layer
regressor.add(LSTM(units=360, return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.2))

#1st hidden layer
regressor.add(LSTM(units=360,return_sequences=True))
regressor.add(Dropout(0.2))

#2nd hidden layer
regressor.add(LSTM(units=360))
regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units=10))

regressor.compile(optimizer='Adam', loss='mean_squared_error')

regressor.fit(x=X_train, y=y_train, batch_size=32,epochs=20,verbose=1)

X_test_2w=X_test_2w.reshape((X_test_2w.shape[0]),(X_test_2w.shape[1]),1)

real_stock_price_2w=y_test_2w
predicted_stock_price_2w=regressor.predict(X_test_2w)

real_stock_price_2w=sc.inverse_transform(real_stock_price_2w)
predicted_stock_price_2w=sc.inverse_transform(predicted_stock_price_2w)
real_stock_price_2w=real_stock_price_2w.transpose()
predicted_stock_price_2w=predicted_stock_price_2w.transpose()


#plotting the results for one week

plt.plot(real_stock_price_2w, color='red',label='Real Stock Price')
plt.plot(predicted_stock_price_2w, color='blue',label='Predicted Stock Price')
plt.title('FB Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price (US$)')
plt.ylim(0,300)
plt.legend()
plt.show()


"""
The idea of this step is taking a total dataset (train+test) creating an input that corresponde to the 
size of 90 before the first day of the test set first day that goes until the end of the dataset 
and predicting week by week of the the test set

"""

data_test = pd.read_csv('MacroTrends_Data_Download_FB_test.csv')
open_stock_price_test=data_test['open'].values

#load again to have it without scailing
open_stock_price_train=data_train['open'].values

total_stock_price = np.concatenate((open_stock_price_train,open_stock_price_test),axis=0)




inputs=total_stock_price[total_stock_price.size - open_stock_price_test.size - 90:]
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)




X_test=[]
for i in range (90, inputs.size,10):
    X_test.append(inputs[i-90:i])

X_test=np.array(X_test)

predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)
predicted_stock_price=predicted_stock_price.flatten()


real_stock_price=open_stock_price_test

plt.plot(real_stock_price, color='red',label='Real Stock Price')
plt.plot(predicted_stock_price, color='blue',label='Predicted Stock Price')
plt.title('FB Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price (US$)')
plt.ylim(0,300)
plt.legend()
plt.show()


























