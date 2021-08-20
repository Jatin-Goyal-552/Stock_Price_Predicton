from django.shortcuts import render
import yfinance as yf
import math
from sklearn.preprocessing import MinMaxScaler
# Create your views here.
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization,LSTM
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
stocks='BTC-INR'
def home(request):
    global stocks
    if request.method == 'POST':
        company = request.POST.get('company1')
        print('company')
        if company=='0':
            stocks='BTC-INR'
        elif company=="1":
            stocks='AAPL'
        elif company=="2":
            stocks="GOOG"


    bitcoin = yf.Ticker(stocks)
    df=bitcoin.history(start='2001-01-19', end='2022-08-13', actions=False)
    # try:
    x=list(map(str,df.index.strftime('%d-%m-%y')))
    # except:
    #     x=list(map(str,df.index))
    y=list(df['High'])
    print("length x",len(x),"length y",len(y))
    df=df.drop(['Open','High','Volume','Low'],axis=1)
    min_max_scalar=MinMaxScaler(feature_range=(0,1))
    data=df.values
    scaled_data=min_max_scalar.fit_transform(data)
    print("len of scaled data",len(scaled_data))
    train_data=scaled_data[:,:]
    x_train=[]
    y_train=[]
    interval=60
    for i in range(interval,len(train_data)):
        x_train.append(train_data[i-interval:i,0])
        y_train.append(train_data[i,0])
    print("len x train",len(x_train),"len y train",len(y_train))
    x_train,y_train=np.array(x_train),np.array(y_train)
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    print("x_train.shape",x_train.shape)
    # model=Sequential()
    # model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    # model.add(LSTM(units=50))
    # model.add(Dense(50))
    # model.add(Dense(1))
    # model.compile(optimizer="adam",loss="mean_squared_error")
    # history=model.fit(x_train,y_train,batch_size=64,epochs=20)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=5)
    df_test=bitcoin.history(start='2001-01-19', end='2022-05-13', actions=False)
    df_test=df_test.drop(['Open','High','Volume','Low'],axis=1)
    predicted=[]
    for i in range(5):
        if predicted!=[]:
            test_value=df_test[-60+i:].values
            test_value=np.append(test_value,predicted)
        else:
            test_value=df_test[-60+i:].values
        print("test_value",test_value)
        # if predicted!=[]:
        #     test_value=min_max_scalar.transform(test_value)+min_max_scalar.transform(np.array(predicted).reshape(-1,1))
        # else:
        #     test_value=min_max_scalar.transform(test_value)
        test=[]
        test.append(test_value)
        # test.extend(predicted)
        test=np.array(test)
        test=np.reshape(test,(test.shape[0],test.shape[1],1))
        tomorrow_prediction=model.predict(test)
        tomorrow_prediction=min_max_scalar.inverse_transform(tomorrow_prediction)
        print("tomorrow_prediction",tomorrow_prediction)
        predicted.append(tomorrow_prediction[0][0])
    # test_value=min_max_scalar.transform(test_value)
    # test=[]
    # test.append(test_value)
    # test=np.array(test)
    # test=np.reshape(test,(test.shape[0],test.shape[1],1))
    # tomorrow_prediction=model.predict(test)
    # tomorrow_prediction=min_max_scalar.inverse_transform(tomorrow_prediction)
    # print("tomorrow_prediction",tomorrow_prediction)
    context={
        'x':x,
        'y':y,
        'company':stocks,
        'predicted_x':[1,2,3,4,5],
        'predicted_y':predicted
    }
    # context={
    #     'x':[1,2,3,4,5],
    #     'y':[5,2,9,1,7]
    # }
    # print(x,y)
    
    return render(request,'home.html',context)