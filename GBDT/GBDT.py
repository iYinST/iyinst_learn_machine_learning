from GBDT.CART_for_GBDT import CART
from Utils.utils import se_loss, gini_loss, sigmoid
import numpy as np


class GBDT:
    def __init__(self, objective='regression', max_tree = 3, max_depth=5, min_samples_leaf=2, min_impurity_decrease=0.,learning_rate = 1e-3):
        self.objective = objective
        if self.objective == 'regression':
            self.loss = se_loss
            self.leaf_weight = np.mean
            self.model_init_func = np.mean
            self.response_gene = lambda pred,y: pred - y
            self.pred_func = lambda x: x
        elif self.objective == 'classification':
            self.loss = gini_loss
            self.leaf_weight = lambda y, pred: (y - pred) / (pred * (1 - pred))
            self.model_init_func = lambda y: - np.log( len(y) / np.sum(y) - 1)
            self.response_gene = lambda pred,y: y - sigmoid(pred)
            self.pred_func = lambda x: np.where(x > .5, 1, 0)

        self.model_init = None
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.depth = 1
        self.max_tree = max_tree
        self.tree_list = []
        self.tree_num = 0
        self.model = None

    def fit(self, X, y):
        if self.model is None:
            self.model_init = self.model_init_func(y)
            self.model = np.repeat(self.model_init, len(y))
        for tree_num in range(self.max_tree):
            # 计算响应值
            response =  self.response_gene(self.model, y)

            # 构建CART树
            new_cart = CART(objective=self.objective, max_depth=self.max_depth, real_label = y)
            new_cart.fit(X,response)
            f = new_cart.predict(X)

            # 添加到list
            self.model += f
            self.tree_list.append(new_cart)

    def predict(self,X):
        predict = self.model_init
        for tree in self.tree_list:
            predict += tree.predict(X)
        return self.pred_func(predict)

if __name__ == '__main__':
    X = np.array([[5,20],[7,30],[21,70],[30,60]])
    y = np.array([0,0,1,1])
    gbdt = GBDT(max_depth=3,max_tree=5,objective = 'classification')
    gbdt.fit(X, y)
    print(gbdt.predict([[25,65]]))
    pass