from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Partially taken from AIF360

class DisparateImpactRemover():
    
    def __init__(self, X_train, X_test, y_train, y_test, sens_attr_name,  clf = LogisticRegression(max_iter=1000)):
        self.columns = X_train.columns
        self.X_train = X_train.to_numpy().tolist().copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.to_numpy().tolist().copy()
        self.y_test = y_test.copy()
        self.sens_attr_name = sens_attr_name
        self.clf = clf
        
    def repair_data(self, X, repair_level = 1.0, kdd=False):
        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")
        for i in self.sens_attr_name:
            self.sens_index = self.columns.get_loc(i)
        repairer = Repairer(X, self.sens_index, repair_level, kdd)
        repaired_data = repairer.repair(X)
        repaired_data_df = pd.DataFrame(repaired_data)
        return repaired_data_df
    
    def fit(self, repair_level = 1.0, kdd=False):
        self.repaired_train = self.repair_data(self.X_train,repair_level=repair_level, kdd=kdd).drop(self.sens_index, axis=1)
        self.repaired_test = self.repair_data(self.X_test, repair_level=repair_level, kdd=kdd).drop(self.sens_index, axis=1)
        #drop sensitive attribute before fitting
        X_train = self.repaired_train
        self.clf.fit(X_train, self.y_train)
        return self
    
    def predict(self):
        X_test = self.repaired_test
        y_pred = self.clf.predict(X_test)
        return y_pred
        