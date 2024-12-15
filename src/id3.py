import numpy as np
import math

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def discretize(self, X, y):        
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        candidates = []
        for i in range(len(X_sorted) - 1):
            if y_sorted[i] != y_sorted[i + 1]:
                midpoint = (X_sorted[i] + X_sorted[i + 1]) / 2
                candidates.append(midpoint)
        
        if not candidates:
            return X
        
        best_gain = -1
        best_threshold = None
        
        for threshold in candidates:
            binary_split = (X >= threshold).astype(int)
            gain = self.information_gain(binary_split, y)
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return (X >= best_threshold).astype(int)

    def information_gain(self, X, y):
        parent_entropy = self.entropy(y)
        unique_values = np.unique(X)
        
        weighted_entropy = 0
        for value in unique_values:
            subset_mask = X == value
            subset_y = y[subset_mask]
            subset_prob = len(subset_y) / len(y)
            subset_entropy = self.entropy(subset_y)
            weighted_entropy += subset_prob * subset_entropy
        
        information_gain = parent_entropy - weighted_entropy
        return information_gain

    def _build_tree(self, X, y, depth=0):
        if (self.max_depth is not None and depth >= self.max_depth) or (len(np.unique(y)) == 1):    
            unique, counts = np.unique(y, return_counts=True) 
            return unique[np.argmax(counts)] # Ambil kelas mayoritas
        
        n_features = X.shape[1]
        best_gain = -1
        best_feature = None
        
        for feature in range(n_features):
            feature_values = X[:, feature] # Ambil semua row feature
            
            # Cek kalau fitur continuous. Ambil batas kalau jumlah row dalam fitur >10 itu continuous
            if len(np.unique(feature_values)) > 10: 
                discretized = self.discretize(feature_values, y) # Discretize
                gain = self.information_gain(discretized, y)
            else:
                gain = self.information_gain(feature_values, y)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        if best_gain == 0:
            unique, counts = np.unique(y, return_counts=True)
            return unique[np.argmax(counts)]
        
        node = {}
        node['feature'] = best_feature
        unique_values = np.unique(X[:, best_feature])

        # Buat tree untuk setiap unique value 
        for value in unique_values:
            mask = X[:, best_feature] == value
            
            sub_X = X[mask]
            sub_y = y[mask]
            
            subtree = self._build_tree(sub_X, sub_y, depth + 1)
            
            if 'branches' not in node:
                node['branches'] = {}
            node['branches'][value] = subtree
        
        return node

    def fit(self, X, y):
        # X = np.array(X)
        # y = np.array(y)
        
        self.tree = self._build_tree(X, y)
        return self

    def predict(self, X):
        # X = np.array(X)
        
        predictions = [self._predict_sample(sample) for sample in X]
        return np.array(predictions)

    def _predict_sample(self, sample):
        node = self.tree
        
        while isinstance(node, dict):
            feature = node['feature']
            feature_value = sample[feature]
            
            if feature_value in node['branches']:
                node = node['branches'][feature_value]
            else:
                break

        # if isinstance(node, dict):
        #     unique, counts = np.unique(list(node['branches'].values()), return_counts=True)
        #     node = unique[np.argmax(counts)]
        
        return node