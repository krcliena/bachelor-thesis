from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression

""" Creates Wrapper around GridSearch reductions approach from Fairlearn
    Copyright (c) Microsoft Corporation and Fairlearn contributors. 
    Licensed under the MIT License.
    Corresponding repository https://github.com/fairlearn/fairlearn/tree/main/fairlearn
    """

class Grid_Search():
    """Parameters
    estimator (estimator) – An estimator implementing methods fit(X, y, sample_weight) 
    and predict(X), where X is the matrix of features, y is the vector of labels 
    (binary classification) or continuous values (regression), and sample_weight is a vector of weights. 
    In binary classification labels y and predictions returned by predict(X) are either 0 or 1. 
    In regression values y and predictions are continuous.
    
    constraints (fairlearn.reductions.Moment) – The disparity constraints expressed as moments
    
    selection_rule (str) – Specifies the procedure for selecting the best model 
    found by the grid search. At the present time, the only valid value is “tradeoff_optimization” 
    which minimizes a weighted sum of the error rate and constraint violation.
    
    constraint_weight (float) – When the selection_rule is “tradeoff_optimization” 
    this specifies the relative weight put on the constraint violation when selecting 
    the best model. The weight placed on the error rate will be 1-constraint_weight
    
    grid_size (int) – The number of Lagrange multipliers to generate in the grid
    
    grid_limit (float) – The largest Lagrange multiplier to generate. 
    The grid will contain values distributed between -grid_limit and grid_limit by default
    
    grid_offset (pandas.DataFrame) – Shifts the grid of Lagrangian multiplier 
    by that value. It is ‘0’ by default
    
    grid – Instead of supplying a size and limit for the grid, users may specify 
    the exact set of Lagrange multipliers they desire using this argument.
    
    sample_weight_name (str) – Name of the argument to estimator.fit() which 
    supplies the sample weights (defaults to sample_weight)      
    """    
    def __init__(self, sens_name,constraints, estimator=LogisticRegression(solver = 'liblinear', fit_intercept=True), 
                 grid_size=10):
        self.estimator = estimator
        self.constraints = constraints 
        self.grid_size = grid_size
        self.sens_name = sens_name
        
    def fit(self, X_train, y_train, S_train):
        X_train = X_train.drop(labels = self.sens_name, axis=1)
        estimator = self.estimator
        grid_size = self.grid_size
        
        if self.constraints == "demographic parity":
            self.model = GridSearch(estimator, DemographicParity(),
                                    grid_size=grid_size)
        if self.constraints == "equalized odds":
            self.model = GridSearch(estimator, EqualizedOdds(),
                                    grid_size=grid_size)
        self.model.fit(X_train, y_train, sensitive_features=S_train)
        #self.predictors = self.model.predictors_
        return self.model
    
    def predict(self, X_test):
        X_test = X_test.drop(labels = self.sens_name, axis=1)
        y_pred = self.model.predict(X_test)
        return y_pred
        
    
    
