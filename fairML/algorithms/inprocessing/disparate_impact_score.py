import fairML.algorithms.inprocessing.fair_classification.main as main 
import fairML.algorithms.inprocessing.fair_classification.utils as ut
import numpy as np
import pandas as pd

class DisparateImpactScore():
"""Reference: Zafar, Muhammad Bilal, et al. "Fairness constraints: Mechanisms for fair classification." Artificial Intelligence and Statistics. PMLR, 2017.
   Creates Wrapper around Zafar et al.'s code.
   """
    
    def __init__(self):
        return None
    
    def fit(self, clstype, datatype = None, const_type = None, dataset = None, sensitive = None):
    """ Parameters
        ----------
        sensitive (str): Name of sensitive attribute if dataset includes multiple sensitive attributes. If not, None is given.
        clstype (str): 'lin-svm' for SVM classifier or 'log-reg' for Logistic Regression
        dataset (dataset_obj): One of the five dataset objects.
        """
        
        self.w = main.execute_multiple(clstype, iterations=1, const_type=const_type, 
                                       datatype = datatype, dataset = dataset, sensitive = sensitive)
        return self
    
    def predict(self, X_test):
    """ Returns predicted outcome of fairness-enhanced classifier."""
 
        X_test = ut.add_intercept(X_test)
        y_pred = np.sign(np.dot(X_test, self.w))
        y_pred = pd.Series(y_pred).replace({-1: 0})
        return y_pred
