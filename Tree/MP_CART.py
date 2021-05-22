import multiprocessing as mp
from functools import reduce

from Utils.utils import *


class Node:
    def __init__(self, value=None, left=None, right=None, instances_index=None):
        self.value = value
        self.left = left
        self.right = right
        self.instances_index = instances_index
        self.split_feature = None
        self.split_point = None



class quickCART:
    def __init__(self, objective='regression', max_depth=10, min_samples_leaf=2, min_impurity_decrease=0.):
        self.objective = objective
        if self.objective == 'regression':
            self.loss = mse_loss
            self.leaf_weight = np.mean
        elif self.objective == 'classification':
            self.loss = gini_loss
            self.leaf_weight = lambda y: np.argmax(np.bincount(y))

        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.root = Node()
        self.min_samples_leaf = min_samples_leaf
        self.depth = 1

        num_cores = int(mp.cpu_count())
        self.pool = mp.Pool(num_cores)

    @time_count
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num_samples = len(self.y)
        self.num_features = self.X.shape[1]
        self.root.instances_index = list(range(X.shape[0]))
        self._generate_node(self.root, self.depth)

    # @time_count
    def _generate_node(self, root: Node, depth: int):

        # 大于最大深度剪枝
        self.depth = max(depth, self.depth)
        if depth >= self.max_depth:
            root.value = self.leaf_weight(self.y[root.instances_index])
            return

        def _get_feature_min_loss_split(root, feature_index):
            min_loss = self.loss(self.y[root.instances_index])
            split_candidates = sorted(np.unique(self.X[root.instances_index, feature_index]))

            def _get_all_split_points_loss(candidate):
                left = [i for i in root.instances_index if self.X[i, feature_index] <= candidate]
                right = [i for i in root.instances_index if self.X[i, feature_index] > candidate]
                # 小于最小样本数剪枝
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    return -1,-1
                # 计算分裂后的loss
                split_loss = (len(left) * self.loss(self.y[left]) + len(right) * self.loss(
                    self.y[right])) / len(root.instances_index)
                return candidate, split_loss

            all_split_points_loss = map(_get_all_split_points_loss, split_candidates)
            all_split_points_loss = filter(lambda x: x[1] > 0, all_split_points_loss)
            min_loss_tuple = reduce(lambda x,y: x if x[1] < y[1] else y, all_split_points_loss,(-1,min_loss))

            return (feature_index, min_loss_tuple[0], min_loss_tuple[1])

        feature_min_loss = [self.pool.apply_async(_get_feature_min_loss_split, args=(root, feature_index)).get() for feature_index in range(self.num_features)]
        min_loss_tuple = reduce(lambda x,y: x if x[2] < y[2] else y,feature_min_loss)

        if min_loss_tuple[1] == -1:
            # 不分裂
            root.value = self.leaf_weight(self.y[root.instances_index])
        else:
            # 分裂
            root.split_point = min_loss_tuple[1]
            root.split_feature = min_loss_tuple[0]
            root.left = Node()
            root.right = Node()

            root.left.instances_index = [i for i in root.instances_index if self.X[i][root.split_feature] <= root.split_point]
            root.right.instances_index = [i for i in root.instances_index if self.X[i][root.split_feature] > root.split_point]
            root.instances_index = None

            self._generate_node(root.left, depth + 1)
            self._generate_node(root.right, depth + 1)

    # @time_count
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
    # f()
    # a_new_decorator(f)
    pass