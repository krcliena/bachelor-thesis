from sklearn.linear_model import LogisticRegression
import fairlearn.postprocessing as postproc
import fairlearn.reductions as red


"""Create Wrapper around Equalized of Opportunity approach from Hardt et al. using fairlearn's implementation of Equality of Opportunity.
   Reference: Hardt, Moritz, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning." 
   Advances in neural information processing systems 29 (2016) 
   """

class EqualityOfOpportunity():
    """Parameters
    estimator (estimator object implementing 'predict' and possibly 'fit') – An estimator whose output is postprocessed.
    
    constraints (str, default='demographic_parity') –
    
    Fairness constraints under which threshold optimization is performed. Possible inputs are:
    
    ’demographic_parity’, ‘selection_rate_parity’ (synonymous)
    match the selection rate across groups
    
    ’{false,true}_{positive,negative}_rate_parity’
    match the named metric across groups
    
    ’equalized_odds’
    match true positive and false positive rates across groups
    
    objective (str, default='accuracy_score') –
    
    Performance objective under which threshold optimization is performed. 
    Not all objectives are allowed for all types of constraints. Possible inputs are:
    
    ’accuracy_score’, ‘balanced_accuracy_score’
    allowed for all constraint types
    
    ’selection_rate’, ‘true_positive_rate’, ‘true_negative_rate’,
    allowed for all constraint types except ‘equalized_odds’
    
    grid_size (int, default=1000) – The values of the constraint metric are 
    discretized according to the grid of the specified size over the interval 
    [0,1] and the optimization is performed with respect to the constraints achieving 
    those values. In case of ‘equalized_odds’ the constraint metric is the false positive rate.
    
    flip (bool, default=False) – If True, then allow flipping the decision if it improves the resulting
    
    prefit (bool, default=False) – If True, avoid refitting the given estimator. 
    Note that when used with sklearn.model_selection.cross_val_score(), 
    sklearn.model_selection.GridSearchCV, this will result in an error. In that case, please use prefit=False.    
    """    
    def __init__(self, estimator=LogisticRegression(solver = 'liblinear', fit_intercept = True), constraints='demographic_parity', 
                 objective='accuracy_score', grid_size=1000, flip=False, prefit=False):
        
        self.model = postproc.ThresholdOptimizer(estimator = estimator, constraints = constraints,
                                                 objective = objective,
                                                 grid_size = grid_size, flip = flip, prefit = prefit)
        
    def fit(self, X_train, y_train, S_train):
        self.model.fit(X_train, y_train, sensitive_features=S_train)
        return self
    
    def predict(self, X_test, sensitive_features, random_state = None):
        y_pred = self.model.predict(X_test, sensitive_features = sensitive_features, random_state = random_state)
        return y_pred
        
    
    def plot_threshold_optimizer(self, ax=None, show_plot=True):
        postproc.plot_threshold_optimizer(self.model, ax = ax, show_plot = show_plot)
