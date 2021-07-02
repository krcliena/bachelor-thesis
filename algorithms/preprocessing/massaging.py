# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:56:51 2021

@author: Karine Louis
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
# =====================================================================================
# Instead of passing X_train, y_train, etc. to the function, should I pass Dataset obj?
# =====================================================================================
class Massaging():
    
    #Do I need init func?
    #Include specific format for Datasets, sensitive array has to be binary!
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, S_train: np.ndarray,
                 sens_group_name, non_sens_group_name, clf = LogisticRegression()):
        self.X_train = X_train
        self.y_train = y_train
        self.S_train = S_train
        self.sens_group_name = sens_group_name
        self.non_sens_group_name = non_sens_group_name
        self.clf = clf
        
        #Create binary numerical sensitive array:
        if not isinstance(sens_group_name, int):
            sens_numerical_arr = S_train.replace({sens_group_name: 1, non_sens_group_name : 0})
        self.sens_numerical_arr = sens_numerical_arr
        
    def calculate_m(self):
        #from Social Data Science Home Assgn 3
        #using m = e * ((n_sensitive * n_nonsensitive) / (n)), 
        # e = r_pos_nonsens - r_pos_sens
        #1. Calculate number of sensitive versus non-sensitive group members
        n_sens = len(self.sens_numerical_arr[self.sens_numerical_arr == 1])
        n_non_sens = len(self.sens_numerical_arr[self.sens_numerical_arr == 0])
        
        #Using dataframe is less code?
        n_pos_sens = 0
        n_pos_non_sens = 0
        for i in range(len(self.S_train)):
            if self.sens_numerical_arr[i] == 1 and self.y_train[i] == 1:
                n_pos_sens = n_pos_sens + 1
            if self.sens_numerical_arr[i] == 0 and self.y_train[i] == 1:
                n_pos_non_sens = n_pos_non_sens + 1
                
        r_pos_sens = n_pos_sens / n_sens
        r_pos_non_sens = n_pos_non_sens / n_non_sens
        e = r_pos_non_sens - r_pos_sens
        m = e * ((n_sens * n_non_sens) / (n_sens + n_non_sens))
        
        self.m = round(m)
        return round(m)
    
    def repair_training_data(self) -> np.ndarray:
        #np.copy(self.y_train)
        y_msg = self.y_train.copy()
        self.clf.fit(self.X_train, self.y_train)
        #predict_proba returns the probability that an individual belongs to classes 0 and 1,
        #here we're only interested in the probability it belongs to the + class
        #We assume that higher scores indicate a higher chance to be in the positive class
        y_score = self.clf.predict_proba(self.X_train)[:,1]
        #female applicants with - in descending order (unprivileged)
        #male applicants with + in ascending order (privileged)
        scores_sens = np.zeros(shape = (1,2))
        scores_non_sens = np.zeros(shape = (1,2))
        #calculate scores in ascending and descending orders resp.
        for i in range(len(y_score)):
            #unpriv group, sensitive_features = 1 means person belongs to sensitive group
            if self.sens_numerical_arr[i] == 1 and self.y_train[i] == 0:
                scores_sens = np.append(scores_sens, [[y_score[i], i]], axis = 0)
            if self.sens_numerical_arr[i] == 0 and self.y_train[i] == 1:
                scores_non_sens = np.append(scores_non_sens, [[y_score[i], i]], axis = 0)
        #delete first array-tuple [0,0]
        scores_sens = np.delete(scores_sens, 0, axis = 0)
        scores_non_sens = np.delete(scores_non_sens, 0, axis = 0)
        
        scores_sens = scores_sens.tolist()
        scores_non_sens = scores_non_sens.tolist()
        
        # 4. rank M_0 in descending order
        scores_sens.sort(reverse = True)
        # 5. rank M_1 in ascending order
        scores_non_sens.sort()
        
        for i in range(self.calculate_m()):
            y_msg[int(scores_sens[i][1])] = 1
            y_msg[int(scores_non_sens[i][1])] = 0
        self.y_msg = y_msg
        return y_msg
    
    #new clf?
    def fit(self, clf_new = LogisticRegression()):
        y_msg = self.repair_training_data()
        #Is that correct?
        self.clf = clf_new
        self.clf.fit(self.X_train, y_msg)
        return self
    
    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        return y_pred
    
