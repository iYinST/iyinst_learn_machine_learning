from sklearn import datasets  # 导入库
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score, precision_score, recall_score
from Tree.CART import CART
from sklearn import tree
import time

# 回归
boston = datasets.load_boston()  # 导入波士顿房价数据
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state= 32)
max_depth = 5

clf = tree.DecisionTreeRegressor(max_depth=max_depth)
t = time.time()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(mse(y_pred, y_test), time.time() - t)

cart = CART(objective='regression',max_depth=max_depth)
t = time.time()
cart.fit(X_train, y_train)
y_pred = cart.predict(X_test)
print(mse(y_pred, y_test), time.time() - t)

# 分类
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=32)
max_depth = 5

clf = tree.DecisionTreeClassifier(max_depth=max_depth)
t = time.time()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(precision_score(y_pred, y_test, average='micro'), recall_score(y_pred, y_test, average='micro'),
      f1_score(y_pred, y_test, average='micro'), time.time() - t)

cart = CART(objective='classification', max_depth=max_depth)
t = time.time()
cart.fit(X_train, y_train)
y_pred = cart.predict(X_test)
print(precision_score(y_pred, y_test, average='micro'), recall_score(y_pred, y_test, average='micro'),
      f1_score(y_pred, y_test, average='micro'), time.time() - t)