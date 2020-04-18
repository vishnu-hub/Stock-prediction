
import numpy as np
from datetime import datetime
import smtplib
import time
from selenium import webdriver
import os
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data


def getStocks():
    #Navigating to the Yahoo stock screener
    driver = webdriver.Chrome(executable_path=r'/home/vishnu/Documents/inQbe/stock/chromedriver')
    url = "https://finance.yahoo.com/screener/predefined/aggressive_small_caps?offset=0&count=202"
    driver.get(url)

    #Creating a stock list and iterating through the ticker names on the stock screener list
    stock_list = []
    ticker = driver.find_element_by_xpath('//*[@id="scr-res-table"]/div[1]/table/tbody/tr[1]/td[1]/a')
    stock_list.append(ticker.text)
    driver.quit()
    
    #Using the stock list to predict the future price of the stock a specificed amount of days
    #print("Tommorow prediction")
    predictData('SWN', 1,'Open',0 )
    predictData('SWN', 1,'High',1 )
    predictData('SWN', 1,'Low',2 )
    predictData('SWN', 1,'Close',3 )

 

def predictData(stock, days, what, i):
   

    start = datetime(2017, 1, 1)
    end = datetime.now()

    #Outputting the Historical data into a .csv for later use
    df = yf.download(stock, start =start, end=end )
    df = df[[what]]

    forecast_out = days
    df['Prediction'] = df[[what]].shift(-forecast_out)

    x = np.array(df.drop(['Prediction'],1))
    x = x[:-forecast_out]

    y = np.array(df['Prediction'])
    y = y[:-forecast_out]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

    svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
    svr_rbf.fit(x_train,y_train)

    #DataFlair - Get score
    svr_rbf_confidence=svr_rbf.score(x_test,y_test)
    #print(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")

    #DataFlair - Create Linear Regression model and train it
    lr=LinearRegression()
    lr.fit(x_train,y_train)

    #DataFlair - Get score for Linear Regression
    lr_confidence=lr.score(x_test,y_test)
    #print(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:] 

    lr_prediction = lr.predict(x_forecast)
    #print("Prediction using Linear Regression")
    #print(lr_prediction)

    svm_prediction = svr_rbf.predict(x_forecast)
    #print("Prediction using SVM")
    #print(svm_prediction)
    if(lr_confidence>=svr_rbf_confidence):
    	abc[i] = lr_prediction
    else:
    	abc[i] = svm_prediction

    


    

def sendMessage(text):
    # If you're using Gmail to send the message, you might need to 
    # go into the security settings of your email account and 
    # enable the "Allow less secure apps" option 
    username = " "
    password = " "

    vtext = " @vtext.com"
    message = text

    msg = """From: %s
    To: %s
    %s""" % (username, vtext, message)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(username, password)
    server.sendmail(username, vtext, msg)
    server.quit()

    print('Sent')

  


if __name__ == '__main__':

    abc = [1,2,3,4]
    getStocks()
    print("Opening Value")
    print(abc[0])
    print("Maximum Value")
    print(abc[1])
    print("Closing Value")
    print(abc[2])
    print("Minimum Value")
    print(abc[3])

    #if(abc[0] > abc[2]):
    	#print("Tommorow will be an up candlestick")


    #else:
    	#print("Tommorow will be a down candlestick")
