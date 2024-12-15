class DecisionTree:
    def __init__(self, max_depth: int = None):
        self.max_depth = max_depth
        self.tree = None

    def split_data(self, x, y, feature_index, threshold):
        """
        Split berdasarkan index fitur dan threshold
        """
        left_idx = [i for i in range(len(x)) if x[i][feature_index] <= threshold]
        right_idx = [i for i in range(len(x)) if x[i][feature_index] > threshold]

        left = ([x[i] for i in left_idx], [y[i] for i in left_idx])
        right = ([x[i] for i in right_idx], [y[i] for i in right_idx])

        return left, right

    def calculate_gini(self, groups, classes):
        """
        Menghitung Gini index untuk split
        """
        total_samples = sum(len(group[1]) for group in groups)
        gini = 0.0
        for x_group, y_group in groups:
            size = len(y_group)
            if size == 0:  # Avoid division by zero
                continue
            score = 0.0
            for class_val in classes:
                proportion = y_group.count(class_val) / size
                score += proportion ** 2
            gini += (1.0 - score) * (size / total_samples)
        return gini

    def best_split(self, x, y):
        """
        Cari index fitur dan threshold terbaik untuk dilakukan split
        """
        classes = list(set(y))
        best_index, best_value, best_score, best_groups = None, None, float('inf'), None
        for feature_index in range(len(x[0])):
            for row in x:
                groups = self.split_data(x, y, feature_index, row[feature_index])
                gini = self.calculate_gini(groups, classes)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = feature_index, row[feature_index], gini, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def to_terminal(self, y):
        """
        Buat terminal node (leaf node)
        """
        return max(set(y), key=y.count)

    def split(self, node, depth):
        """
        Split node secara sekursif untuk membuat tree
        """
        left, right = node['groups']
        del node['groups']

        if not left[1] or not right[1]:
            node['left'] = node['right'] = self.to_terminal(left[1] + right[1])
            return

        if self.max_depth and depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left[1]), self.to_terminal(right[1])
            return

        node['left'] = self.best_split(*left)
        self.split(node['left'], depth + 1)

        node['right'] = self.best_split(*right)
        self.split(node['right'], depth + 1)

    def fit(self, x_train, y_train):
        self.tree = self.best_split(x_train, y_train)
        self.split(self.tree, 1)

    def predict_point(self, row, node):
        """
        Predict kelas 1 poin data
        """
        if isinstance(node, dict):
            if row[node['index']] <= node['value']:
                return self.predict_point(row, node['left'])
            else:
                return self.predict_point(row, node['right'])
        else:
            return node

    def predict(self, x_pred):
        """
        Predict kelas semua data
        """
        return [self.predict_point(row, self.tree) for row in x_pred]

# Contoh
# x_train = [[2.771244718, 1.784783929], [1.728571309, 1.169761413], [3.678319846, 2.81281357],
#           [3.961043357, 2.61995032], [2.999208922, 2.209014212]]
# y_train = ["apel", "apel", "oren", "oren", "oren"]
# x_test = [[2.771244718, 1.784783929], [3.678319846, 2.81281357]]

# model = DecisionTree(max_depth=3)
# model.fit(x_train, y_train)
# predictions = model.predict(x_test)
# print(predictions)
