from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Partially taken from AIF360 under the Apache License Version 2.0, January 2004 http://www.apache.org/licenses/
#Corresponding repository https://github.com/Trusted-AI/AIF360/tree/master/aif360


class DisparateImpactRemover():
"""References:
        .. [1] M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and
           S. Venkatasubramanian, "Certifying and removing disparate impact."
           ACM SIGKDD International Conference on Knowledge Discovery and Data
           Mining, 2015.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, sens_attr_name,  clf = LogisticRegression(max_iter=1000)):
    """
        Parameters
        ----------
            repair_level (float): Repair amount. 0.0 is no repair while 1.0 is
                full repair.
            sens_attr_name (str): Single sensitive attribute with which to
                do repair.
            clf: Classifier to use for training, default is LogisticRegression()
        """
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
    """
        Repair data then train on the repaired data using classifier clf. 
        """
        self.repaired_train = self.repair_data(self.X_train,repair_level=repair_level, kdd=kdd).drop(self.sens_index, axis=1)
        self.repaired_test = self.repair_data(self.X_test, repair_level=repair_level, kdd=kdd).drop(self.sens_index, axis=1)
        #drop sensitive attribute before fitting
        X_train = self.repaired_train
        self.clf.fit(X_train, self.y_train)
        return self
    
    def predict(self):
    """
        Returns
        -------
        y_pred: Predicted outcomes after doing fairness-enhancment
        """
        X_test = self.repaired_test
        y_pred = self.clf.predict(X_test)
        return y_pred
        
