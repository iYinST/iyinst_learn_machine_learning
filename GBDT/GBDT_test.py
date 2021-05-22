from sklearn import datasets  # 导入库
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score, precision_score, recall_score
from GBDT.GBDT import GBDT
from sklearn import tree
import time

# 回归
boston = datasets.load_boston()  # 导入波士顿房价数据
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state= 32)
max_depth = 5

clf = tree.DecisionTreeRegressor(max_depth=max_depth)
clf = clf.fit(X_train, y_train)
t = time.time()
y_pred = clf.predict(X_test)
print(mse(y_pred, y_test), time.time() - t)

gbdt = GBDT(objective='regression',max_depth=max_depth)
t = time.time()
gbdt.fit(X_train, y_train)
y_pred = gbdt.predict(X_test)
print(mse(y_pred, y_test), time.time() - t)

# 分类
cancer = datasets.load_breast_cancer()  # 导入乳腺癌数据
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=32)
max_depth = 3

clf = tree.DecisionTreeClassifier(max_depth=max_depth)
t = time.time()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(precision_score(y_pred, y_test), recall_score(y_pred, y_test),
      f1_score(y_pred, y_test), time.time() - t)

gbdt = GBDT(objective='classification',max_depth=max_depth)
t = time.time()
gbdt.fit(X_train, y_train)
y_pred = gbdt.predict(X_test)
print(precision_score(y_pred, y_test), recall_score(y_pred, y_test),
      f1_score(y_pred, y_test), time.time() - t)