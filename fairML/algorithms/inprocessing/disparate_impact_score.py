#Classification classes need to be {-1,1} instead of {0,1}
#Wrapper class for Zafar et al

import fair.algorithms.inprocessing.fair_classification.main as main 
import fair.algorithms.inprocessing.fair_classification.utils as ut
import numpy as np
import pandas as pd

class DisparateImpactScore():
    
    #give already splitted data or give dataframe and split here?
    def __init__(self):
        return None
    
    def fit(self, clstype, datatype = None, const_type = None, dataset = None, sensitive = None):
        
        self.w = main.execute_multiple(clstype, iterations=1, const_type=const_type, 
                                       datatype = datatype, dataset = dataset, sensitive = sensitive)
        return self
    
    def predict(self, X_test):
        X_test = ut.add_intercept(X_test)
        y_pred = np.sign(np.dot(X_test, self.w))
        y_pred = pd.Series(y_pred).replace({-1: 0})
        return y_pred