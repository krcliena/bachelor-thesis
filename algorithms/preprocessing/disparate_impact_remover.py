from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Partially taken from AIF360

class DisparateImpactRemover():
    
    def __init__(self, X_train, X_test, y_train, y_test, sens_attr_name):
        self.columns = X_train.columns
        self.X_train = X_train.to_numpy().tolist().copy()
        self.y_train = y_train.to_numpy().copy()
        self.X_test = X_test.to_numpy().tolist().copy()
        self.y_test = y_test.to_numpy().copy()
        self.sens_attr_name = sens_attr_name
    
    def repair_data(self, repair_level = 1.0, kdd=False):
        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")
        self.sens_index = len(self.X_train[0]) - 1
        self.repairer = Repairer(self.X_train, self.sens_index, repair_level, kdd)
        self.repaired_train = self.repairer.repair(self.X_train)
        self.repaired_test = self.repairer.repair(self.X_test)
        repaired_train_df = pd.DataFrame(self.repaired_train)
        repaired_test_df = pd.DataFrame(self.repaired_test)
        return repaired_train_df, repaired_test_df
    
    def fit(self, clf = LogisticRegression(), repair_level = 1.0, kdd=False):
        self.clf = clf
        self.repaired_train, self.repaired_test = self.repair_data(repair_level=repair_level, kdd=kdd)
        #drop sensitive attribute before fitting
        X_train = self.repaired_train.drop(self.sens_index, axis=1)
        self.clf.fit(X_train, self.y_train)
        return self
    
    def predict(self):
        X_test = self.repaired_test.drop(self.sens_index, axis=1)
        y_pred = self.clf.predict(X_test)
        return y_pred
        