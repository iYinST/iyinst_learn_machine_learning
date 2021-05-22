import numpy as np
from Utils.utils import se_loss


class Node:
    def __init__(self, value=None, left=None, right=None, instances_index=None):
        self.value = value
        self.left = left
        self.right = right
        self.instances_index = instances_index
        self.split_feature = None
        self.split_point = None


class CART:
    def __init__(self, objective='regression', max_depth=10, min_samples_leaf=1, min_impurity_decrease=0., real_label = None):
        self.objective = objective
        if self.objective == 'regression':
            self.loss = se_loss
            self.leaf_weight = lambda res,label: -np.mean(res)
        elif self.objective == 'classification':
            self.loss = se_loss
            self.leaf_weight = lambda res, label : np.sum(res) / np.sum((label - res) * (1 - label + res))


        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.root = Node()
        self.min_samples_leaf = min_samples_leaf
        self.depth = 1
        self.real_label = real_label

    # @time_count
    def fit(self, X, y):
        self.root.instances_index = list(range(X.shape[0]))
        self._generate_node(self.root, X, y, self.depth)

    def _generate_node(self, root: Node, X: np.array, y: np.array, depth: int):

        # 大于最大深度剪枝
        self.depth = max(depth, self.depth)
        if depth >= self.max_depth:
            root.value = self.leaf_weight(y[root.instances_index], self.real_label[root.instances_index])
            return

        split_feature, split_point = -1, -1
        min_loss = self.loss(y[root.instances_index])

        # 寻找分裂点
        for feature_index in range(X.shape[1]):
            split_candidate = sorted(np.unique(X[root.instances_index, feature_index]))
            for candidate in split_candidate:
                left = [i for i in root.instances_index if X[i, feature_index] <= candidate]
                right = [i for i in root.instances_index if X[i, feature_index] > candidate]

                # 小于最小样本数剪枝
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                # 计算分裂后的loss
                split_loss = self.loss(y[left]) + self.loss(y[right])

                # 更新loss
                if split_loss < min_loss and self.loss(y[root.instances_index]) - split_loss > self.min_impurity_decrease:
                    min_loss = split_loss
                    split_feature = feature_index
                    split_point = candidate

        if split_point == -1:
            # 不分裂
            root.value = self.leaf_weight(y[root.instances_index], self.real_label[root.instances_index])
        else:
            # 分裂
            root.split_point = split_point
            root.split_feature = split_feature
            root.left = Node()
            root.right = Node()

            root.left.instances_index = [i for i in root.instances_index if X[i][split_feature] <= split_point]
            root.right.instances_index = [i for i in root.instances_index if X[i][split_feature] > split_point]
            root.instances_index = None

            self._generate_node(root.left, X, y, depth + 1)
            self._generate_node(root.right, X, y, depth + 1)

    def predict(self, X):
        result = np.zeros([len(X)])
        for item, x in enumerate(X):
            root = self.root
            while root.value is None:
                if x[root.split_feature] <= root.split_point:
                    root = root.left
                else:
                    root = root.right
            result[item] = root.value
        return result


if __name__ == '__main__':
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    cart = CART()
    cart.fit(X, y)
    print(cart.predict([[3]]))
    pass
