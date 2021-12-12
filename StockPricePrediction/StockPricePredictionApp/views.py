from django.shortcuts import render,HttpResponse
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
import csv
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import optimizers
import pandas as pd
from datetime import timedelta, date

df=None
df1=None
df2=None
def home(request):
    global df
    # stocks="AAPL"
    # start_date='2000-06-19'
    # close_date='2022-08-13'
    context={"flag":False}
    if request.method == 'POST':
        stocks = str(request.POST.get('company1'))
        start_date = str(request.POST.get('start_date'))
        close_date= str(request.POST.get('close_date'))
        print("dwwwwwwwwwwwwwww")
        print(start_date,close_date)
        print('company')
        # if company=='0':
        #     stocks='BTC-USD'
        # elif company=="1":
        #     stocks='AAPL'
        # elif company=="2":
        #     stocks="GOOG"
        print("stocks",stocks)

        bitcoin = yf.Ticker(stocks)
        des=bitcoin.info
        temp_des={}
        for key in des:
            if des[key]!='None' and des[key]!=[]:
                temp_des[key]=des[key]
        des=temp_des
        print("description",des)
        # print(des)
        # print("description",type(des))
        # print("news",bitcoin.news)
        df=bitcoin.history(start=str(start_date), end=str(close_date), actions=False)
        print("df",df.head())
        print("***********************")
        print(bitcoin.institutional_holders)
        print("major stakeholder",bitcoin.recommendations)
        df['Date']=df.index.strftime('%d-%m-%Y')
        # try:
        x=list(map(str,df.index.strftime('%d-%m-%y')))
        # except:
        #     x=list(map(str,df.index))
        y_high=list(df['High'])
        y_open=list(df['Open'])
        y_low=list(df['Low'])
        y_close=list(df['Close'])
        y_volume=list(df['Volume'])
        # print("length x",len(x),"length y",len(y))
        # df=df.drop(['Open','High','Volume','Low'],axis=1)
        # min_max_scalar=MinMaxScaler(feature_range=(0,1))
        # data=df.values
        # scaled_data=min_max_scalar.fit_transform(data)
        # print("len of scaled data",len(scaled_data))
        # train_data=scaled_data[:,:]
        # x_train=[]
        # y_train=[]
        # interval=60
        # for i in range(interval,len(train_data)):
        #     x_train.append(train_data[i-interval:i,0])
        #     y_train.append(train_data[i,0])
        # print("len x train",len(x_train),"len y train",len(y_train))
        # x_train,y_train=np.array(x_train),np.array(y_train)
        # x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        # print("x_train.shape",x_train.shape)
    
        # model = Sequential()
        # model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        # model.add(LSTM(64, return_sequences=False))
        # model.add(Dense(25))
        # model.add(Dense(1))

        # # Compile the model
        # model.compile(optimizer='adam', loss='mean_squared_error')

        # # Train the model
        # model.fit(x_train, y_train, batch_size=128, epochs=1)
        # df_test=bitcoin.history(start='2001-01-19', end='2022-05-13', actions=False)
        # df_test=df_test.drop(['Open','High','Volume','Low'],axis=1)
        # predicted=[]
        # for i in range(5):
        #     if predicted!=[]:
        #         test_value=df_test[-60+i:].values
        #         test_value=np.append(test_value,predicted)
        #     else:
        #         test_value=df_test[-60+i:].values
        #     print("test_value",test_value)
            
        #     test=[]
        #     test.append(test_value)
        #     test=np.array(test)
        #     test=np.reshape(test,(test.shape[0],test.shape[1],1))
        #     tomorrow_prediction=model.predict(test)
        #     tomorrow_prediction=min_max_scalar.inverse_transform(tomorrow_prediction)
        #     print("tomorrow_prediction",tomorrow_prediction)
        #     predicted.append(tomorrow_prediction[0][0])
    
        context={
            'x':x,
            'y_high':y_high,
            'y_low':y_low,
            'y_open':y_open,
            'y_close':y_close,
            'y_volume':y_volume,
            'company':stocks,
            'df':df,
            'predicted_x':[1,2,3,4,5],
            'predicted_y':[5,4,3,2,1],
            'max_price':round(max(y_high),2),
            'min_price':round(min(y_low),2),
            'last_day_price':round(y_close[-1],2),
            'change_in_price':round(y_high[-1]-y_high[0],2),
            'change_in_precentage':round(((y_high[-1]-y_high[0])/y_high[0])*100,2),
            "description":des,
            "flag":True,
            'company':stocks,
            'start_date':start_date,
            'close_date':close_date
        }
    
    return render(request,'home2.html',context)

def compare(request):
    stocks1="BTC-INR"
    stocks2="AAPL"
    start_date='2021-06-19'
    close_date='2022-08-13'
    context={
        "flag":False
    }
    if request.method == 'POST':
        stocks1 = request.POST.get('company1')
        stocks2 = request.POST.get('company2')
        start_date = str(request.POST.get('start_date'))
        close_date= str(request.POST.get('close_date'))
        print(start_date,close_date)
        print('company')
        # if company1=='0':
        #     stocks1='BTC-INR'
        # elif company1=="1":
        #     stocks1='AAPL'
        # elif company1=="2":
        #     stocks1="GOOG"
        # if company2=='0':
        #     stocks2='BTC-INR'
        # elif company2=="1":
        #     stocks2='AAPL'
        # elif company2=="2":
        #     stocks2="GOOG"  
        global df1,df2
        data1 = yf.Ticker(stocks1)
        df1=data1.history(start=str(start_date), end=str(close_date), actions=False)
        df1['Date']=df1.index.strftime('%d-%m-%y')
        print(df1)
        # try:
        x_stock1=list(map(str,df1.index.strftime('%d-%m-%y')))
        # except:
        #     x=list(map(str,df.index))
        y_high_stock1=list(df1['High'])
        y_open_stock1=list(df1['Open'])
        y_low_stock1=list(df1['Low'])
        y_close_stock1=list(df1['Close']) 
        y_volume_stock1=list(df1['Volume'])
        data2 = yf.Ticker(stocks2)
        df2=data2.history(start=str(start_date), end=str(close_date), actions=False)
        df2['Date']=df2.index.strftime('%d-%m-%y')
        print(df2)
        # try:
        x_stock2=list(map(str,df2.index.strftime('%d-%m-%y')))
        # except:
        #     x=list(map(str,df.index))
        y_high_stock2=list(df2['High'])
        y_open_stock2=list(df2['Open'])
        y_low_stock2=list(df2['Low'])
        y_close_stock2=list(df2['Close'])  
        y_volume_stock2=list(df2['Volume'])
        x_final=x_stock2[:]
        if len(x_stock2)<len(x_stock1):
            y_high_stock2=y_high_stock2[-len(x_stock2):]
            y_open_stock2=y_open_stock2[-len(x_stock2):]
            y_low_stock2=y_low_stock2[-len(x_stock2):]
            y_close_stock2=y_close_stock2[-len(x_stock2):]
            y_volume_stock2=y_volume_stock2[-len(x_stock2):]
            x_final=x_stock2[:]
        elif len(x_stock2)>len(x_stock1) :
            y_high_stock1=y_high_stock1[-len(x_stock1):]
            y_open_stock1=y_open_stock1[-len(x_stock1):]
            y_low_stock1=y_low_stock1[-len(x_stock1):]
            y_close_stock1=y_close_stock1[-len(x_stock1):]
            y_volume_stock1=y_volume_stock1[-len(x_stock1):]
            x_final=x_stock1[:]
        context={
            'x':x_final,
            'y_high_stock1':y_high_stock1,
            'y_open_stock1':y_open_stock1,
            'y_low_stock1':y_low_stock1,
            'y_close_stock1':y_close_stock1,
            'y_high_stock2':y_high_stock2,
            'y_open_stock2':y_open_stock2,
            'y_low_stock2':y_low_stock2,
            'y_close_stock2':y_close_stock2,
            'y_volume_stock1':y_volume_stock1,
            'y_volume_stock2':y_volume_stock2,
            'company1':stocks1,
            'company2':stocks2,
            'df1':df1,
            'df2':df2,
            'max_price_stock1':round(max(y_high_stock1),2),
            'min_price_stock1':round(min(y_low_stock1),2),
            'last_day_price_stock1':round(y_close_stock1[-1],2),
            'change_in_price_stock1':round(y_high_stock1[-1]-y_high_stock1[0],2),
            'change_in_precentage_stock1':round(((y_high_stock1[-1]-y_high_stock1[0])/y_high_stock1[0])*100,2),
            'max_price_stock2':round(max(y_high_stock2),2),
            'min_price_stock2':round(min(y_low_stock2),2),
            'last_day_price_stock2':round(y_close_stock2[-1],2),
            'change_in_price_stock2':round(y_high_stock2[-1]-y_high_stock2[0],2),
            'change_in_precentage_stock2':round(((y_high_stock2[-1]-y_high_stock2[0])/y_high_stock2[0])*100,2),
            'flag':True,
            "start_date":start_date,
            "close_date":close_date
        }
    return render(request,'compare2.html',context)


def download(request,id):
    global df,df1,df2
    print(df)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data.csv"' # your filename
    # df.to_csv("data.csv")
    # writer = csv.writer(response)
    # writer.writerow(['Username','Email'])

    # users = User.objects.all().values_list('username','email')

    # for user in users:
    #     writer.writerow(user)
    writer = csv.writer(response)
    writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    if id=='0':
        for ind in df.index:
            writer.writerow([ind,df['Open'][ind],df['High'][ind],df['Low'][ind],df['Close'][ind],df['Volume'][ind]])
    elif id=='1':
        for ind in df1.index:
            writer.writerow([ind,df1['Open'][ind],df1['High'][ind],df1['Low'][ind],df1['Close'][ind],df1['Volume'][ind]])
    elif id=='2':
        for ind in df2.index:
            writer.writerow([ind,df2['Open'][ind],df2['High'][ind],df2['Low'][ind],df2['Close'][ind],df2['Volume'][ind]])
    return response


def predict(request):
    stocks="BTC-INR"
    start_date='2000-04-01'
    close_date='2022-08-13'
    context={"flag":False}
    if request.method == 'POST':
        stocks = request.POST.get('company1')
        days=int(request.POST.get('days'))
        # start_date = str(request.POST.get('start_date'))
        # close_date= str(request.POST.get('close_date'))
        # print("dwwwwwwwwwwwwwww")
        # print(start_date,close_date)
        # print('company')
        # if company=='0':
        #     stocks='BTC-INR'
        # elif company=="1":
        #     stocks='AAPL'
        # elif company=="2":
        #     stocks="GOOG"


        bitcoin = yf.Ticker(stocks)
        df=bitcoin.history(start=str(start_date), end=str(close_date), actions=False)
        print(df)
        print("***********************")
        # print(bitcoin.institutional_holders)
        # df['Date']=df.index.strftime('%d-%m-%y')
        x=list(map(str,df.index.strftime('%d-%m-%y')))
        y_high=list(df['Close'])
        # y_open=list(df['Open'])
        # y_low=list(df['Low'])
        # y_close=list(df['Close'])
        # y_volume=list(df['Volume'])
        # print("length x",len(x),"length y",len(y))
        df=df.drop(['Open','High','Volume','Low'],axis=1)
        min_max_scalar=MinMaxScaler(feature_range=(0,1))
        data=df.values
        scaled_data=min_max_scalar.fit_transform(data)
        print("len of scaled data",len(scaled_data))
        train_data=scaled_data[:,:]
        x_train=[]
        y_train=[]
        interval=90
        for i in range(interval,len(train_data)):
            x_train.append(train_data[i-interval:i,0])
            y_train.append(train_data[i,0])
        print("len x train",len(x_train),"len y train",len(y_train))
        x_train,y_train=np.array(x_train),np.array(y_train)
        x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        print("x_train.shape",x_train.shape)
        # model = prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode="additive")

        stop = EarlyStopping(
        monitor='val_loss', 
        mode='min',
        patience=7
        )

        checkpoint= ModelCheckpoint(
            filepath='./',
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        model=Sequential()
        model.add(LSTM(200,return_sequences=True,input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=100))
        model.add(Dense(100))
        model.add(Dense(1))

        adam = optimizers.Adam(lr=0.0005)

        model.compile(optimizer=adam, loss='mse')
        # checkpoint= ModelCheckpoint(
        #     filepath='./',
        #     save_weights_only=True,
        #     monitor='val_loss',
        #     mode='min',
        #     save_best_only=True)
        # model.load("/")
        model.fit(x_train, y_train, batch_size=128, epochs=50,shuffle=True, validation_split=0.05, callbacks = [checkpoint,stop])
        model.load_weights("./")
        df_test=bitcoin.history(start='2000-01-01', end='2022-05-13', actions=False)
        df_test=df_test.drop(['Open','High','Volume','Low'],axis=1)
        predicted=[]
        for i in range(days):
            if predicted!=[]:
                if (-interval+i)<0:
                    test_value=df_test[-interval+i:].values
                    test_value=np.append(test_value,predicted)
                else:
                    test_value=np.array(predicted)
            else:
                test_value=df_test[-interval+i:].values
            print("test_value",test_value)
            test_value=test_value[-interval:].reshape(-1,1)
            test_value=min_max_scalar.transform(test_value)
            test=[]
            test.append(test_value)
            test=np.array(test)
            test=np.reshape(test,(test.shape[0],test.shape[1],1))
            # print("test",test)
            tomorrow_prediction=model.predict(test)
            tomorrow_prediction=min_max_scalar.inverse_transform(tomorrow_prediction)
            print("tomorrow_prediction",tomorrow_prediction)
            predicted.append(tomorrow_prediction[0][0])
        predicted_x=[]
        for i in range(1,days+1):
            predicted_x.append( str((date.today() + timedelta(days=i)).strftime('%d-%m-%y')))
        if max(predicted)>y_high[-1]:
            buy="Yes"
        else:
            buy="No"
        context={
                'x':x,
                'y_high':y_high,
                'company':stocks,
                'predicted_x':predicted_x,
                'predicted_y':predicted,
                "flag":True,
                "days":days,
                "csv":zip(predicted_x,predicted),
                "max_price":round(max(predicted),2),
                "min_price":round(min(predicted),2),
                "buy":buy,
                "change_in_precentage":round(((max(predicted)-min(predicted))/(min(predicted)))*100,2),
                "change_in_price":round((max(predicted)-min(predicted)),2)
            }
    
    return render(request,'predict.html',context)