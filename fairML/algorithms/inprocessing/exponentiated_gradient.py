from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression

""" Creates Wrapper around ExponentiatedGradient reductions approach from Fairlearn
    Copyright (c) Microsoft Corporation and Fairlearn contributors. 
    Licensed under the MIT License.
    Corresponding repository https://github.com/fairlearn/fairlearn/tree/main/fairlearn
    """

class Exponentiated_Gradient():
    
    """
    Parameters
    estimator (estimator) – An estimator implementing methods fit(X, y, sample_weight) 
    and predict(X), where X is the matrix of features, y is the vector of labels 
    (binary classification) or continuous values (regression), and sample_weight 
    is a vector of weights. In binary classification labels y and predictions 
    returned by predict(X) are either 0 or 1. In regression values y and predictions are continuous.

    constraints (fairlearn.reductions.Moment) – The disparity constraints expressed as moments

    eps (float) – Allowed fairness constraint violation; the solution is 
    guaranteed to have the error within 2*best_gap of the best error under 
    constraint eps; the constraint violation is at most 2*(eps+best_gap)
    
    max_iter (int) – Maximum number of iterations
    
    nu (float) – Convergence threshold for the duality gap, corresponding to a 
    conservative automatic setting based on the statistical uncertainty in 
    measuring classification error
    
    eta_0 (float) – Initial setting of the learning rate
    
    run_linprog_step (bool) – if True each step of exponentiated gradient is 
    followed by the saddle point optimization over the convex hull of classifiers 
    returned so far; default True
    
    sample_weight_name (str) – Name of the argument to estimator.fit() which 
    supplies the sample weights (defaults to sample_weight)
    """
    
    def __init__(self, sens_name, constraints, estimator=LogisticRegression(solver = 'liblinear', fit_intercept=True)):
        self.estimator = estimator
        self.constraints = constraints
        self.sens_name = sens_name
        
    def fit(self, X_train, y_train, S_train):
        X_train = X_train.drop(labels = self.sens_name, axis=1)
        estimator = self.estimator
        #constraints = self.constraints
        if self.constraints == "equalized odds":
            self.model = ExponentiatedGradient(estimator, EqualizedOdds())
        if self.constraints == "demographic parity":
            self.model = ExponentiatedGradient(estimator, DemographicParity())
        self.model.fit(X_train, y_train, sensitive_features=S_train)
        self.predictors = self.model.predictors_
        return self.model
    
    def predict(self, X_test):
        X_test = X_test.drop(labels=self.sens_name, axis=1)
        y_pred = self.model.predict(X_test)
        return y_pred
