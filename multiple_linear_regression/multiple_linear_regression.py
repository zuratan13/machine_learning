# coding: shift-jis
import requests, zipfile
from io import StringIO
import io
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
res=requests.get(url).content

auto = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None)

auto.columns = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels',
                'engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders',
                'engine-size','fuel-system','bore','stroke','compression-ration','horsepower','peak-rpm', 'city-mpg','highway-mpg','price']

print('�f�[�^�̊m�F:{}'.format(auto.shape))

print(auto.head())

auto = auto[['price','horsepower','width','height']]
print(auto.isin(['?']).sum())

auto = auto.replace('?', np.nan).dropna()

print('�f�[�^�̊m�F:{}'.format(auto.shape))

print('�f�[�^�^�̊m�F(�ύX�O)\n{}\n'.format(auto.dtypes))

auto = auto.assign(price=pd.to_numeric(auto.price))
auto = auto.assign(horsepower=pd.to_numeric(auto.horsepower))

print('�f�[�^�^�̊m�F(�ύX��)\n{}\n'.format(auto.dtypes))


X = auto.drop('price', axis=1)
y = auto['price']

X = X.values
y = y.values


X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Predicting the Test set results
y_pred = model.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print('����W��((train):{:.3f}'.format(model.score(X_train, y_train)))
print('����W��((test):{:.3f}'.format(model.score(X_test, y_test)))

print('\n��A�W��\n{}'.format(model.coef_))
print('�ؕ�:{:.3f}'.format(model.intercept_))

