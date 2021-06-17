# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:56:51 2021

@author: Rafik
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math

class Massaging():
    
    #Do I need init func?
    #def __init__(self, X, y):
    
    def calculate_m(self,):
        return self
    
    def fit(self, X: np.ndarray, sensitive_features: np.ndarray, 
                      y: np.ndarray, m:int, clf = LogisticRegression()) -> np.ndarray:
        
        y_msg = np.copy(y)
        clf.fit(X,y)
        #predict_proba returns the probability that an individual belongs to classes 0 and 1,
        #here we're only interested in the probability it belongs to the + class
        #We assume that higher scores indicate a higher chance to be in the positive class
        y_score = clf.predict_proba(X)[:,1]
        #female applicants with - in descending order (unprivileged)
        #male applicants with + in ascending order (privileged)
        scores_sens = np.zeros(shape = (1,2))
        scores_unsens = np.zeros(shape = (1,2))
        #calculate scores in ascending and descending orders resp.
        for i in range(len(y_score)):
            #unpriv group, sensitive_features = 1 means person belongs to sensitive group
            if sensitive_features[i] == 1 and y[i] == 0:
                scores_sens = np.append(scores_sens, [[y_score[i], i]], axis = 0)
            if sensitive_features[i] == 0 and y[i] == 1:
                scores_unsens = np.append(scores_unsens, [[y_score[i], i]], axis = 0)
        #delete first array-tuple [0,0]
        scores_sens = np.delete(scores_sens, 0, axis = 0)
        scores_unsens = np.delete(scores_unsens, 0, axis = 0)
        
        scores_sens = scores_sens.tolist()
        scores_unsens = scores_unsens.tolist() 
        
        # 4. rank M_0 in descending order
        scores_sens.sort(reverse = True)
        
        # 5. rank M_1 in ascending order
        scores_unsens.sort()
        
        for i in range(m):
            y_msg[int(scores_sens[i][1])] = 1
            y_msg[int(scores_unsens[i][1])] = 0
        self.y_msg = y_msg
        
        return self
    
    def transform(self,):
        return self
