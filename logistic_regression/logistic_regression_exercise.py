import requests, zipfile
from io import StringIO
import io
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score

cancer = load_breast_cancer()

print(cancer.data)
print(cancer.target)

print(len(cancer.data[0]))

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

print('正解率:{:.3f}'.format(model.score(X_train,y_train)))
print('正解率:{:.3f}'.format(model.score(X_test,y_test)))




