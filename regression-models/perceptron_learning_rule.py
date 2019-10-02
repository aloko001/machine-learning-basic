import numpy as np

#import pandas as pd
#import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow import backend

class perceptron_rule(object):
    """Perceptron_rule_classifier.

    Parameters
    -----------
    eta: float
        Learning rate ranges from 0.0 to 1.0
    n_iter: int 
        Passes over the training dataset for evaluation
    random_state: int
        Random number seed generator for generating random weight.
        initialization.

    Attributes
    -----------
    weight_: 1d-array (x1)
        weights after fitting.
    errors: list
        evaluates number of misclassifications (updates) in each epoch

    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        -----------
        x = {array}, shape = [n_labels, n_features]
            Training vectors, where n_labels is the number of
            labels and
            n_features is the number of features.
        y = {array, shape = [n_labels]

        Returns
        -------
        self : class(object)

        """
        random_gen = np.random.RandomState(self.random_state)
        self.weight_ = random_gen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x1, target in zip(X, y):
                update = self.eta * (target - self.predict(x1))
                self.weight_[1:] += update * x1
                self.weight_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        return np.dot(X, self.weight_[1:]) + self.weight_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.1, 1, -1)


