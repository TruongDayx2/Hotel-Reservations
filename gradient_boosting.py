import numpy as np
from decision_tree import DecisionTreeRegressor, DecisionTreeClassifier

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.mean = None

    def fit(self, X, y):
        print("Training Gradient Boosting Regressor")
        # Calculate the mean of y
        self.mean = np.mean(y)
        # Initialize the prediction with the mean of y
        self.prediction = np.full(len(X), self.mean)
        for i in range(self.n_estimators):
            # Calculate the gradient of the loss function use cross-entropy
            gradient = -1 * (y - self.prediction)
            # Train a decision tree on the negative gradient
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            print("Training tree", i+1)
            tree.fit(X, gradient)
            # Multiply the tree output with the learning rate and add it to the prediction
            self.prediction += self.learning_rate * tree.predict(X)
            # Add the trained tree to the ensemble
            self.models.append(tree)

class GradientBoostingClassifier(): # use cross-entropy and sigmoid
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.mean = None
    
    def fit(self, X, y):
        print("Training Gradient Boosting Classifier")
        # Calculate the mean of y
        self.mean = np.mean(y)
        # Initialize the prediction with the mean of y
        self.prediction = np.full(len(X), self.mean)
        for i in range(self.n_estimators):
            # Calculate the gradient of the loss function use cross-entropy
            gradient = -1 * (y - self.prediction)
            # Train a decision tree on the negative gradient
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            print("Training tree", i+1)
            tree.fit(X, gradient)
            # Multiply the tree output with the learning rate and add it to the prediction
            self.prediction += self.learning_rate * tree.predict(X)
            # Add the trained tree to the ensemble
            self.models.append(tree)
        
    def predict(self, X):
        # Initialize the prediction with the mean of y
        self.prediction = np.full(len(X), self.mean)
        for i in range(self.n_estimators):
            # Multiply the tree output with the learning rate and add it to the prediction
            self.prediction += self.learning_rate * self.models[i].predict(X)
        # Apply the sigmoid function to the prediction
        self.prediction = 1 / (1 + np.exp(-self.prediction))
        # Return the class with the highest probability
        return np.round(self.prediction)