import numpy as np
from sklearn.linear_model import LogisticRegression

class Reweighing():
"""References:
        Kamiran, Faisal, and Toon Calders. "Data preprocessing techniques 
        for classification without discrimination." Knowledge and Information Systems 33.1 (2012):
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, S_train: np.ndarray,
                 sens_group_name, non_sens_group_name, clf = LogisticRegression(max_iter = 1000)):
    """Parameters
        ----------
            sens_group_name: Name of privileged group 
            non_sens_group_name: Name of unprivileged group
            clf: Classifier to use for training, default is LogisticRegression()
        """
        self.X_train = X_train
        self.y_train = y_train
        self.S_train = S_train
        self.sens_group_name = sens_group_name
        self.non_sens_group_name = non_sens_group_name
        self.clf = clf
        
        #Create binary numerical sensitive array:
        if not isinstance(sens_group_name, int):
            sens_numerical_arr = S_train.replace({sens_group_name: 0, non_sens_group_name : 1})
        self.sens_numerical_arr = sens_numerical_arr
        
    def reweighing(self):
    """Calculate weights for each sensitive attribute value and class label combination
    
        Returns
        -------
        weights (list): Weights matrix for each combination
        """
        n = len(self.sens_numerical_arr)
        weights = [0 for _ in range(n)]
        S = [0, 0]
        C = [0, 0]
        S_C = [[0, 0], [0, 0]]
        W = [[0, 0], [0, 0]]
    
        for i in range(n):
            sens_attr = int(self.sens_numerical_arr[i])
            cls = int(self.y_train[i])
            S[sens_attr] += 1
            C[cls] += 1
            S_C[sens_attr][cls] += 1
    
        for s in [0, 1]:
            for c in [0, 1]:
                W[s][c] = (S[s] * C[c])/(n * S_C[s][c])
    
        for i in range(n):
            sens_attr = int(self.sens_numerical_arr[i])
            cls = int(self.y_train[i])
            weights[i] = W[sens_attr][cls]
    
        return W, S_C, weights
    
    def fit(self):
    """
        Use calculated weights to train. Must include classifier which supports weights.
        """
        self.W, self.S_C, self.weights = self.reweighing()
        weights = self.weights
        #Is that correct?
        self.clf.fit(self.X_train, self.y_train, sample_weight = weights)
        return self
    
    def predict(self, X_test):
      """
        Returns
        -------
        y_pred: Predicted outcomes after doing fairness-enhancment
        """
        y_pred = self.clf.predict(X_test)
        return y_pred
    
    
