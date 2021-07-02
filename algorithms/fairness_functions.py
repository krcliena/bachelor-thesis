import numpy as np
import fair.algorithms.inprocessing.fair_classification.utils as ut
import fair.algorithms.inprocessing.fair_classification.loss_funcs as lf
import fair.algorithms.inprocessing.fair_classification.funcs_disp_mist as fdm
from copy import deepcopy

def disparate_impact_score_n_fold(data):
    '''
    This implementation typically performs a 10 fold cross-validation with the entire data
    :param data:
    :return:
    '''
    NUM_FOLDS = 10

    X = np.copy(data.tr_dt)
    y = np.copy(data.tr_c)
    x_control = {'s1': np.copy(data.tr_s)}

    ut.compute_p_rule(x_control["s1"], y)

    apply_fairness_constraints = 0
    apply_accuracy_constraint = 0
    sep_constraint = 0

    loss_function = lf._logistic_loss
    #loss_function = lf.linear_SVM_loss
    X = ut.add_intercept(X)

    test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, \
    cov_dict_train_arr = ut.compute_cross_validation_error(X, y, x_control, NUM_FOLDS, loss_function, apply_fairness_constraints,
                                                           apply_accuracy_constraint, sep_constraint, ['s1'],
                                                           [{} for i in range(0, NUM_FOLDS)])


    print ("== Unconstrained (original) classifier ==")
    ut.print_classifier_fairness_stats(test_acc_arr, correlation_dict_test_arr, cov_dict_test_arr, "s1")


    """ Now classify such that we achieve perfect fairness """

    apply_fairness_constraints = 1
    cov_factor = 0
    test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, \
    cov_dict_train_arr = ut.compute_cross_validation_error(X, y, x_control, NUM_FOLDS, loss_function, apply_fairness_constraints,
                                                           apply_accuracy_constraint, sep_constraint, ['s1'], [{'s1':cov_factor} for i in range(0,NUM_FOLDS)])
    print ("== Constrained (fair) classifier ==")
    ut.print_classifier_fairness_stats(test_acc_arr, correlation_dict_test_arr, cov_dict_test_arr, "s1")


def disparate_impact_score_test_train(data, clstype, dtype):
    #Changed so I can incorporate my data objects
    '''
    This splits the data into (train+val) and test so as to make comparisons with the proposed model
    :param data:
    :return:
    '''
    if dtype == 1:
        X = {'train': np.copy(data.tr_dt), 'test': np.copy(data.ts_dt)}
        y = {'train': np.copy(data.tr_c), 'test': np.copy(data.ts_c)}
        x_control = {'train': {'s1': np.copy(data.tr_s)}, 'test': {'s1': np.copy(data.ts_s)}}
    
    if dtype == 2:
        y_train = data.y_train.replace({0 : -1})
        y_test = data.y_test.replace({0: -1})
        
        X = {'train': np.copy(data.X_train.to_numpy()), 'test': np.copy(data.X_test.to_numpy())}
        y = {'train': np.copy(y_train.to_numpy()), 'test': np.copy(y_test.to_numpy())}
        x_control = {'train': {'s1': np.copy(data.S_train.to_numpy())}, 'test': {'s1': np.copy(data.S_test.to_numpy())}}
        
    ut.compute_p_rule(x_control['train']["s1"], y['train'])

    apply_fairness_constraints = 0
    apply_accuracy_constraint = 0
    sep_constraint = 0

    if clstype == 'lin-svm':
        loss_function = lf.linear_SVM_loss
    elif clstype == 'log-reg':
        loss_function = lf._logistic_loss
    else:
        pass

    #print(X['train'][0])
    X['train'] = ut.add_intercept(X['train'])
    X['test'] = ut.add_intercept(X['test'])
    #print(X['train'][0])

    test_score, train_score, correlation_dict_test, correlation_dict_train, \
    cov_dict_test, cov_dict_train, w = ut.compute_single_fold(X, y, x_control, loss_function, apply_fairness_constraints,
                                                           apply_accuracy_constraint, sep_constraint,
                                                           ['s1'], {})

    ut.print_classifier_fairness_stats_single_fold(test_score, correlation_dict_test, "s1")

    """ Now classify such that we achieve perfect fairness """
    
    print("Now classify such that we achieve perfect fairness")
    apply_fairness_constraints = 1
    cov_factor = 0
    test_score, train_score, correlation_dict_test, correlation_dict_train, \
    cov_dict_test, cov_dict_train, w_f = ut.compute_single_fold(X, y, x_control, loss_function, apply_fairness_constraints,
                                                           apply_accuracy_constraint, sep_constraint,
                                                           ['s1'], {'s1': cov_factor})

    acc_, fair_ = ut.print_classifier_fairness_stats_single_fold(test_score, correlation_dict_test, "s1")

    return acc_, fair_, w_f


def disparate_mistreatment_score(data, dtype, const_type=1):
    
    if dtype == 1:
        X = {'train': np.copy(data.tr_dt), 'test': np.copy(data.ts_dt)}
        y = {'train': np.copy(data.tr_c), 'test': np.copy(data.ts_c)}
        x_control = {'train': {'s1': np.copy(data.tr_s)}, 'test': {'s1': np.copy(data.ts_s)}}
    
    if dtype == 2:
        y_train = data.y_train.replace({0 : -1})
        y_test = data.y_test.replace({0: -1})
        
        X = {'train': np.copy(data.X_train.to_numpy()), 'test': np.copy(data.X_test.to_numpy())}
        y = {'train': np.copy(y_train.to_numpy()), 'test': np.copy(y_test.to_numpy())}
        x_control = {'train': {'s1': np.copy(data.S_train.to_numpy())}, 'test': {'s1': np.copy(data.S_test.to_numpy())}}
        
    cons_params = None  # constraint parameters, will use them later
    loss_function = "linear_svm" #"logreg"  # perform the experiments with logistic regression
    EPS = 1e-4

    sensitive_attrs = ['s1']

    def train_test_classifier():
        w = fdm.train_model_disp_mist(X['train'], y['train'], x_control['train'], loss_function, EPS, cons_params)

        train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(
            w, X['train'], y['train'], x_control['train'], X['test'], y['test'], x_control['test'], sensitive_attrs)

        # accuracy and FPR are for the test because we need of for plotting
        # the covariance is for train, because we need it for setting the thresholds
        return w, test_score, s_attr_to_fp_fn_test, cov_all_train

    print("== Unconstrained (original) classifier ==")
    w_uncons, acc_uncons, s_attr_to_fp_fn_test_uncons, cov_all_train_uncons = train_test_classifier()


    print ("== Classifier with fairness constraint ==")

    it = 0.05
    mult_range = np.arange(1.0, 0.0 - it, -it).tolist()

    #print(mult_range)

    acc_arr = []
    fpr_per_group = {0: [], 1: []}
    fnr_per_group = {0: [], 1: []}

    cons_type = const_type  # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
    tau = 5.0
    mu = 1.2

    for m in mult_range:
        sensitive_attrs_to_cov_thresh = deepcopy(cov_all_train_uncons)
        for s_attr in sensitive_attrs_to_cov_thresh.keys():
            for cov_type in sensitive_attrs_to_cov_thresh[s_attr].keys():
                for s_val in sensitive_attrs_to_cov_thresh[s_attr][cov_type]:
                    sensitive_attrs_to_cov_thresh[s_attr][cov_type][s_val] *= m

        cons_params = {"cons_type": cons_type,
                       "tau": tau,
                       "mu": mu,
                       "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

        w_cons, acc_cons, s_attr_to_fp_fn_test_cons, cov_all_train_cons = train_test_classifier()

        fpr_per_group[0].append(s_attr_to_fp_fn_test_cons["s1"][0.0]["fpr"])
        fpr_per_group[1].append(s_attr_to_fp_fn_test_cons["s1"][1.0]["fpr"])
        fnr_per_group[0].append(s_attr_to_fp_fn_test_cons["s1"][0.0]["fnr"])
        fnr_per_group[1].append(s_attr_to_fp_fn_test_cons["s1"][1.0]["fnr"])

        acc_arr.append(acc_cons)
