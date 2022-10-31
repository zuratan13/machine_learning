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
from sklearn.metrics import confusion_matrix, accuracy_score

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'

res=requests.get(url).content

mushroom = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None)

mushroom.columns = ['classes','cap_shape','cap_surface','cap_color','odor','bruises',
            'gill_attachment','gill_spacing','gill_size','gill_color','stalk_shape',
            'stalk_root','stalk_surface_above_ring','stalk_surface_below_ring',
            'stalk_color_above_ring','stalk_color_below_ring','veil_type','veil_color',
            'ring_number','ring_type','spore_print_color','population','habitat']

print(mushroom.head())

mushroom_dummy = pd.get_dummies(mushroom[['gill_color','gill_attachment','odor','cap_color']])
print(mushroom_dummy.head())

mushroom_dummy['flg'] = mushroom['classes'].map(lambda x: 1 if x=='p' else 0)

X = mushroom_dummy.drop('flg', axis=1)
y = mushroom_dummy['flg']

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=0)

model = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('正解率(train):{:.3f}'.format(model.score(X_train,y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test,y_test)))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

dot_data=StringIO()
tree.export_graphviz(model, out_file=dot_data)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.progs = {'dot': u"C:\\Program Files\\Graphviz\\bin\\dot.exe"}
file_name = u"C:\\Users\\matsuo\\Pictures\\FaceRetrieve\\DecisionTree.png"
graph.write_png(file_name)



