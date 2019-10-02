import pandas as pd
import os, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

os.chdir('C:\\Users\\USER\\Desktop')
df = pd.read_csv('all.csv')

df = df[['Bullish','Neutral','Bearish','S&P 500 Weekly High','S&P 500 Weekly Low','S&P Weekly Close','Total']]
df['High_change_wk'] = (df['S&P 500 Weekly High'] - df['S&P 500 Weekly Close']) / df['S&P 500 Weekly Close'] * 100.0
df['Low_change_wk'] = (df['S&P 500 Weekly Low'] - df['S&P 500 Weekly Close']) / df['S&P 500 Weekly Close'] * 100.0
df['Bul_Neu_Bea'] = (df['Bullish'] + df['Neutral'] + df['Bearish']) * 100.0

df = df[['High_change_wk','Low_change_wk','Bul_Neu_Bea','S&P 500 Weekly Close','Total']]

forecast_col = 'S&P 500 Weekly Close'
df.fillna(-99999, inplace=True)

forecast_out int(math.ceil(0.01*len(df)))

df['Label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['Label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['Label'])
y = np.array(df['Label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan


last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['S&P 500 Weekly Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





