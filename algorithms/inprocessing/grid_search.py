import fairlearn.reductions as red 
from sklearn.linear_model import LogisticRegression

""" Create Wrapper around GridSearch approach from Agarwal et al."""

class GridSearch():
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
    def __init__(self, estimator=LogisticRegression(), constraints=red.DemographicParity(),
                 selection_rule='tradeoff_optimization', constraint_weight=0.5, 
                 grid_size=10, grid_limit=2.0, grid_offset=None, grid=None, 
                 sample_weight_name='sample_weight'):
        
        self.model = red.GridSearch(estimator, constraints, selection_rule, constraint_weight,
                                    grid_size, grid_limit, grid_offset, grid, sample_weight_name)
        
    def fit(self, X_train, y_train, S_train):
        self.model.fit(X_train, y_train, sensitive_features=S_train)
        self.predictors = self.model.predictors_
        return self
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
        
    
    