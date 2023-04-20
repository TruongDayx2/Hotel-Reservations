import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  
        self.threshold = threshold  
        self.left = left  
        self.right = right  
        self.value = value  

# class DecisionTreeRegressor:
#     def __init__(self, max_depth=None, min_samples_split=2):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.root = None

#     def build_tree(self, X, y, depth=0):
#         n_samples, n_features = X.shape
#         n_labels = len(np.unique(y))

#         # stopping criteria
#         if (depth >= self.max_depth
#                 or n_labels == 1
#                 or n_samples < self.min_samples_split):
#             leaf_value = np.mean(y)
#             return Node(value=leaf_value)

#         # search for the best split
#         feature_idxs = np.arange(n_features)
#         best_feature, best_threshold = None, None
#         best_variance_reduction = 0
#         for feature_idx in feature_idxs:
#             thresholds = np.unique(X[:, feature_idx])
#             for threshold in thresholds:
#                 y_left = y[X[:, feature_idx] < threshold]
#                 y_right = y[X[:, feature_idx] >= threshold]
#                 if len(y_left) == 0 or len(y_right) == 0:
#                     continue
#                 variance_reduction = self.variance_reduction(y, y_left, y_right)
#                 if variance_reduction > best_variance_reduction:
#                     best_feature = feature_idx
#                     best_threshold = threshold
#                     best_variance_reduction = variance_reduction

#         # split the data and grow subtrees
#         left_idxs = np.where(X[:, best_feature] < best_threshold)[0]
#         right_idxs = np.where(X[:, best_feature] >= best_threshold)[0]
#         left = self.build_tree(X[left_idxs, :], y[left_idxs], depth=depth+1)
#         right = self.build_tree(X[right_idxs, :], y[right_idxs], depth=depth+1)
#         return Node(best_feature, best_threshold, left, right)

#     def variance_reduction(self, y, y_left, y_right):
#         var_total = np.var(y)
#         var_left = np.var(y_left)
#         var_right = np.var(y_right)
#         n_total = len(y)
#         n_left = len(y_left)
#         n_right = len(y_right)
#         variance_reduction = var_total - (n_left/n_total * var_left + n_right/n_total * var_right)
#         return variance_reduction

#     def fit(self, X, y):

#         print("Building tree with max depth", self.max_depth)
#         self.root = self.build_tree(X, y)

#     def predict(self, X):
#         return np.array([self.traverse_tree(x, self.root) for x in X])

#     def traverse_tree(self, x, node):
#         if node.value is not None:
#             return node.value
#         if x[node.feature_index] < node.threshold:
#             return self.traverse_tree(x, node.left)
#         else:
#             return self.traverse_tree(x, node.right)


class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _build_tree(self, X, y, depth):
        # Termination conditions: max depth or all samples have the same target value
        if depth == self.max_depth or np.all(y == y[0]):
            return y.mean()
        
        # Find the best split for the current node
        split_idx, split_val = self._find_best_split(X, y)
        
        # Split the data into left and right subsets
        left_idxs = np.where(X[:, split_idx] < split_val)
        right_idxs = np.where(X[:, split_idx] >= split_val)
        
        # Recursively build the left and right subtrees
        left_tree = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right_tree = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
        
        # Return a dictionary representing the current node
        return {'split_idx': split_idx,
                'split_val': split_val,
                'left': left_tree,
                'right': right_tree}
    
    def _find_best_split(self, X, y):
        best_idx, best_val, best_loss = None, None, np.inf
        
        # Try splitting at every feature and every unique value
        for i in range(X.shape[1]):
            for val in np.unique(X[:, i]):
                left_idxs = np.where(X[:, i] < val)
                right_idxs = np.where(X[:, i] >= val)
                
                # Skip if either subset is empty
                if len(left_idxs[0]) == 0 or len(right_idxs[0]) == 0:
                    continue
                
                # Calculate the mean squared error for the current split
                left_mse = np.mean((y[left_idxs] - y[left_idxs].mean())**2)
                right_mse = np.mean((y[right_idxs] - y[right_idxs].mean())**2)
                total_mse = left_mse + right_mse
                
                # Update the best split if necessary
                if total_mse < best_loss:
                    best_idx, best_val, best_loss = i, val, total_mse
        
        return best_idx, best_val
    
    def _traverse_tree(self, x, node):
        # If the current node is a leaf, return its value
        if isinstance(node, float):
            return node
        
        # Traverse left or right subtree based on the current feature value
        if x[node['split_idx']] < node['split_val']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])
