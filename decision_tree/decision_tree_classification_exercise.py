import requests, zipfile
from io import StringIO
import io
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
from six import StringIO
from IPython.display import Image,display
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.data)
print(cancer.target)

print(len(cancer.data[0]))

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)

model = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0)
model.fit(X_train,y_train)

print('正解率(train):{:.3f}'.format(model.score(X_train,y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test,y_test)))

dot_data=StringIO()
tree.export_graphviz(model, out_file=dot_data)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.progs = {'dot': u"C:\\Program Files\\Graphviz\\bin\\dot.exe"}
file_name = u"C:\\Users\\matsuo\\Pictures\\FaceRetrieve\\DecisionTree_cancer.png"
graph.write_png(file_name)
