import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble

path_train = 'data/train/train.csv'
path_test = 'data/test/test.csv'

data = pd.read_csv(path_train, low_memory=False)
data_test = pd.read_csv(path_test, low_memory=False)

X = data.values[:60000, 0:-2].tolist()
Y_label = data.values[:60000, -2].tolist()
user_id = data.values[:60000, -1].tolist()
label = []
for i in Y_label:
    if i not in label :
        label.append(i)
Y = []
for j in Y_label:
    for k in range(len(label)):
        if j == label[k]:
            Y.append(k)

scaler = preprocessing.StandardScaler()
train_data, y = shuffle(X, Y)
x_train, x_test, y_train, y_test = train_test_split(train_data, y, test_size=0.3, random_state=0)

test_data = data_test.values[:, 0:-1].tolist()
user_id = data_test.values[:, -1].tolist()


###########3.具体方法选择##########

####3.5随机森林回归####
bdt = ensemble.RandomForestRegressor(n_estimators=100,max_depth=15, min_samples_split=10, min_samples_leaf=3)
bdt.fit(x_train, y_train)
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn.metrics import f1_score

y_pred = bdt.predict(x_test)
scorerf = bdt.score(x_train,y_train)
print('gbt模型得分', scorerf)


pred = bdt.predict(test_data)
print(pred)
result = pred
frame_pred = DataFrame(user_id)
frame_pred['predict'] = result

path3 = 'submit.csv'
print("保存开始")
frame_pred.rename(columns={0: 'user_id', 1: 'predict'}, inplace=True)
frame_pred.to_csv(path3, index=False)
print('保存结束')
