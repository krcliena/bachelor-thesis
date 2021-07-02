import numpy as np
from sklearn.linear_model import LogisticRegression

class Reweighing():
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, S_train: np.ndarray,
                 sens_group_name, non_sens_group_name):
        self.X_train = X_train
        self.y_train = y_train
        self.S_train = S_train
        self.sens_group_name = sens_group_name
        self.non_sens_group_name = non_sens_group_name
        
        #Create binary numerical sensitive array:
        if not isinstance(sens_group_name, int):
            sens_numerical_arr = S_train.replace({sens_group_name: 1, non_sens_group_name : 0})
        self.sens_numerical_arr = sens_numerical_arr
        
    def reweighing(self):
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
    
    #classifier is LogisticRegression, user must be able to choose as they wish
    def fit(self):
        _, _, weights = self.reweighing()
        #Is that correct?
        self.clf = LogisticRegression(class_weight= weights)
        self.clf.fit(self.X_train, self.y_train)
        return self
    
    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        return y_pred
    
    
