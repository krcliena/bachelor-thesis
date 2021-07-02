import fairlearn.reductions as red
from sklearn.linear_model import LogisticRegression

#Is ExponentiatedGradient same as GridSearch ?

class ExponentiatedGradient():
    
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
    
    def __init__(self, estimator=LogisticRegression(), constraints=red.DemographicParity(), 
                 eps=0.01, max_iter=50, nu=None, eta0=2.0, run_linprog_step=True, 
                 sample_weight_name='sample_weight'):
    
        self.model = red.ExponentiatedGradient(estimator, constraints, eps,
                                               max_iter, nu, eta0, run_linprog_step,
                                               sample_weight_name)
        
    def fit(self, X_train, y_train, S_train):
        self.model.fit(X_train, y_train, sensitive_features=S_train)
        self.predictors = self.model.predictors_
        return self
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred