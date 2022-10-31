import requests, zipfile
from io import StringIO
import io
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

res = requests.get(url).content

adult = pd.read_csv(io.StringIO(res.decode('utf-8')), header = None)

adult.columns = ['age','workclass','fnlwgt','education','education-num','marital-status',
                'occupation','relationship','race','sex','capital-gain','capital-loss',
                'hours-per-week','native-country','flg-50K']

print('データの形式:{}'.format(adult.shape))
print('欠損の数:{}'.format(adult.isnull().sum().sum()))

print(adult.head())

print(adult.groupby('flg-50K').size())

adult['fin_flg'] = adult['flg-50K'].map(lambda x: 1 if x == ' >50K' else 0)

print(adult.groupby('fin_flg').size())

X = adult[['age','fnlwgt','education-num','capital-gain','capital-loss']]
y = adult['fin_flg']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=0)

print(X_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

print('正解率(train){:.3f}'.format(model.score(X_train, y_train)))
print('正解率(test){:.3f}'.format(model.score(X_test, y_test)))