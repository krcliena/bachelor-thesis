import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class PreferentialSampling():
"""References:
        Kamiran, Faisal, and Toon Calders. "Data preprocessing techniques 
        for classification without discrimination." Knowledge and Information Systems 33.1 (2012):
    """    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, S_train: np.ndarray,
                 sens_group_name, non_sens_group_name, clf = LogisticRegression(max_iter=1000)):
    """Parameters
        ----------
            sens_group_name: Name of privileged group 
            non_sens_group_name: Name of unprivileged group
            clf: Classifier to use for training, default is LogisticRegression()
        """
        self.X_train = X_train
        self.y_train = y_train
        self.S_train = S_train
        self.sens_group_name = sens_group_name
        self.non_sens_group_name = non_sens_group_name
        self.clf = clf
        
        #Create binary numerical sensitive array:
        if not isinstance(sens_group_name, int):
            sens_numerical_arr = S_train.replace({sens_group_name: 0, non_sens_group_name : 1})
        self.sens_numerical_arr = sens_numerical_arr
        
    def calculate_W(self, s, c):
    """ Calculate weights for different sensitive attribute and class label combinitations.
        Returns
        -------
        w_s_c (float): Weight of class c with sensitive attribute value s
        """
        #W(s,c) := (n_sens = s * n_class = c) / (n * n_sens_class)
        nr_s = len(self.sens_numerical_arr[self.sens_numerical_arr == s])
        nr_c = len(self.y_train[self.y_train==c])
        nr_s_c = 0
        n = len(self.y_train)
        
        for i in range(len(self.y_train)):
            if self.sens_numerical_arr[i] == s and self.y_train[i] == c:
                nr_s_c = nr_s_c + 1
                
        w_s_c = (nr_s * nr_c) / (n * nr_s_c)
        # print("nr_s:", nr_s)
        # print("nr_c:", nr_c)
        # print("nr_s_c:", nr_s_c)
        # print("n:", n)
        # print("w_s_c with s=",s, "and c=", c, w_s_c)
        return w_s_c
    
    def repair_training_data(self) -> np.ndarray:
       """
        Repair training data as definition of Preferential Sampling as follows:
        #1. Add W(sens,pos) copies of sens_pos to ps_arr
        #2. Add W(sens, pos) - (W(sens,pos) * n_sens_pos) lowest ranked elements of sens_pos to ps_arr
        #3. Add W(sens, neg) * n_sens_neg lowest ranked elements of sens_neg to ps_arr
        #4. Add W(non_sens, pos) * n_non_sens_pos highest ranked elements of non_sens_pos to ps_arr
        #5. Add W(non_sens, neg) copies of non_sens_neg to ps_arr
        #6. Add W(non_sens, neg) - (W(sens,neg)*non_sens_neg) highest ranked elements of non_sens_neg to ps_arr
        #7. Return sampled X_train, y_train and A_train
     
        Returns
        -------
        repaired Data (pd.DataFrame)
        """
        
        W = [[0,0],[0,0]]
        for i in [0,1]:
            for l in [0,1]:
                W[i][l] = self.calculate_W(i, l)
        
        self.W = W
        
        predictor = LogisticRegression(max_iter=1000)
        predictor.fit(self.X_train,self.y_train)
        probs = predictor.predict_proba(self.X_train)[:,1]
    
        n = len(self.X_train)
        
        sens_attr = {0: 'D', 1: 'F'}
        cls = {0: 'N', 1: 'P'}

        groups = {'DP': [], 'DN': [], 'FP': [], 'FN': []}

        for i in range(n):
            c = int(self.y_train[i])
            s = int(self.sens_numerical_arr[i])
    
            g_p = sens_attr[s]+cls[c]
        
            groups[g_p].append((i, probs[i]))
            
        #Every group has the indices belonging to this group and their probability
        p_sampled = []
        
        #Line 8 in Algorithm
        inds, probs = zip(*groups['DP'])
        #Discriminated = 0, positive = 1
        for i in range(int(W[0][1])):
            p_sampled.extend(inds)
        print(int(W[0][1]))
        
        #Line 9 in Algorithm
        samp_size = int((W[0][1] - int(W[0][1]))*len(groups['DP']))
        groups['DP'] = sorted(groups['DP'], key=lambda x: x[1])
        inds, probs = zip(*groups['DP'])
        p_sampled.extend(inds[:samp_size])
        print(samp_size)
        
        #Line 10 in Algorithm
        samp_size = int(W[0][0]*len(groups['DN']))
        groups['DN'] = sorted(groups['DN'], key=lambda x: x[1])
        inds, probs = zip(*groups['DN'])
        p_sampled.extend(inds[:samp_size])
        print(samp_size)
        
        #Line 11 in Algorithm
        samp_size = int(W[1][1]*len(groups['FP']))
        groups['FP'] = sorted(groups['FP'], key=lambda x: x[1], reverse=True)
        inds, probs = zip(*groups['FP'])
        p_sampled.extend(inds[:samp_size])
        print(samp_size)
        
        #Line 12 in Algorithm
        inds, probs = zip(*groups['FN'])
        for _ in range(int(W[1][0])):
            p_sampled.extend(inds)
        print(int(W[1][0]))
        #Line 13 in Algorithm
        samp_size = int((W[1][0] - int(W[0][0]))*len(groups['FN']))
        groups['FN'] = sorted(groups['FN'], key=lambda x: x[1], reverse=True)
        inds, probs = zip(*groups['FN'])
        p_sampled.extend(inds[:samp_size])
        self.p_sampled = p_sampled
        print(samp_size)
        
        #p_sampled now includes all the indices of the original dataset to 
        #be added to the repaired
        sampled_data = {'data': [], 'class': [], 'sens_attr': []}
        for ind in p_sampled:
            #print(ind)
            #print(self.y_train[ind])
            sampled_data['data'].append(self.X_train.to_numpy()[ind])
            sampled_data['class'].append(self.y_train[ind])
            sampled_data['sens_attr'].append(self.S_train[ind])
            
        repaired_X_train = pd.DataFrame(sampled_data['data'], columns = self.X_train.columns)
        repaired_A_train = sampled_data['sens_attr']
        repaired_y_train = sampled_data['class']
        self.groups = groups
        self.sampled_data = sampled_data
        self.repaired_X_train = repaired_X_train
        self.repaired_A_train = repaired_A_train
        self.repaired_y_train = repaired_y_train
        return repaired_X_train, repaired_y_train, repaired_A_train


    def fit(self):
    """
        Repair data then train on the repaired data using classifier clf. 
        """
        X_train, y_train, S_train = self.repair_training_data()
        #Is that correct?
        self.clf.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
    """
        Returns
        -------
        y_pred: Predicted outcomes after doing fairness-enhancment
        """
        y_pred = self.clf.predict(X_test)
        return y_pred
    
    
    
    
    

