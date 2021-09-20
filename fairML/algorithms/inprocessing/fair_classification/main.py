from fair.algorithms.inprocessing.fair_classification.generate_data import DataDispMis, DataDispImp, Dataset
from fair.algorithms.inprocessing.fair_classification.fairness_functions import disparate_impact_score_test_train, disparate_mistreatment_score
import math
import numpy as np


#just create a wrapper, create a Class with fit and predict
def get_data(datatype):

    if datatype == 1: # disparate impact
        means = [[2, 2], [-2, -2]]
        covars = [[[5, 1], [1, 5]], [[10, 1], [1, 3]]]
        n_samples = 5000
        disc_factor = math.pi / 4.0
        data = DataDispImp(means, covars, n_samples, disc_factor)

    elif datatype == 2: # disparate mistreatment only FPR
        means = [[2, 2], [2, 2], [-2, -2], [-1, 0]]
        covars = [[[3, 1], [1, 3]], [[3, 1], [1, 3]], [[3, 1], [1, 3]], [[3, 1], [1, 3]]]
        n_samples = 2500
        data = DataDispMis(means, covars, n_samples)

    elif datatype == 3: # disparate mistreatment both FPR, FNR opposite signs
        means = [[2, 0], [2, 3], [-1, -3], [-1, 0]]
        covars = [[[5, 1], [1, 5]], [[5, 1], [1, 5]], [[5, 1], [1, 5]], [[5, 1], [1, 5]]]
        n_samples = 2500
        data = DataDispMis(means, covars, n_samples)

    elif datatype == 4: # disparate mistreatment both FPR, FNR same sign
        means = [[1, 2], [2, 3], [0, -1], [-5, 0]]
        covars = [[[5, 2], [2, 5]], [[10, 1], [1, 4]], [[7, 1], [1, 7]], [[5, 1], [1, 5]]]
        n_samples = 2500
        data = DataDispMis(means, covars, n_samples)

    #elif datatype == 5:
    #    data = Dataset('../Datasets/adult.p')
    #    return data

    else:
        pass

    return data

#the one i have to call, -1, 1 not 0,1
def execute_multiple(clstype, iterations=1, const_type=None, datatype = None, dataset = None, sensitive = None):
    if not datatype == None:
        dtype = 1
        data = get_data(datatype)
    if not dataset == None:
        dtype = 2
        data = dataset

    accrs = []
    fairs = []
    #what does _ mean?
    for _ in range(iterations):
        #Commented this because I don't need it for doing pref. sampling + di score
        # data.train_test_split(train_size=0.80, val=False, sensitive = sensitive)
        #consttype fnr, fpr, etc.
        if const_type is None:
            acc_, fair_, w = disparate_impact_score_test_train(data, clstype, dtype)
        else:
            acc_, fair_ = disparate_mistreatment_score(data, dtype, const_type)

        #print(w) # format -- intercept, coef_...
        accrs.append(acc_)
        fairs.append(fair_)

    print(f'avg_accr: {np.mean(accrs)}, std_accr: {np.std(accrs)}, \
            avg_fair: {np.mean(fairs)}, std_fair: {np.std(fairs)}')
    print("accrs: ", accrs, "fairs: ", fairs)
    return w


def fit(data=None, clstype=None):
    acc_, fair_, w = disparate_impact_score_test_train(data, clstype)
    print(f"accuracy: {acc_}")
    print(f"fairness: {fair_}")
    return w


#if __name__ == '__main__':
    #results = execute_multiple(datatype=1, clstype='log-reg', iterations=1)
    # clstype - lin-svm, log-reg
    #print(results)
    #const_type = 1 # 1-> only FPR, 2-> only FNR, 4-> both FPR and FNR
    #disparate_impact_score_test_train(data, const_type)



