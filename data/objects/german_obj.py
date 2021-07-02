# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:20:19 2021

@author: Karine Louis
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GermanData():
    
    #Export preprocessed data? 
    def __init__(self, scale = True):
        
        #1. Define class attributes belonging to Adult Dataset
        self.df = pd.read_csv(r'C:\Users\Rafik\Desktop\BA\fair-main\fair\data\raw\german_credit.csv')
        self.data_name = 'german'
        self.sensitive_attributes_names = ['sex']
        self.priv_class_name = 'male'
        self.pos_class = 1
        #Is this copy necessary, if I were to use self.df.replace would it be a reference to the obj?
        dataframe = self.df.copy()
        
        #Individuals with "workclass" = "Never-worked" have an occupation of '?',
        #replace these "?" with "No-occupation". Drop rest. 
        #indices = dataframe[dataframe["workclass"] == " Never-worked"].index
        #for index in indices:
        #    dataframe.loc[index, 'occupation'] = ' No-occupation'
        
        #2. Drop missing data, missing label indication: '?'
        #df_dropped = dataframe.replace(' ?', np.nan).dropna()
        
        #3. Define attributes, sensitive and target data
        self.X = dataframe.drop('credit', axis = 1)
        self.target = dataframe["credit"]
        # #1 equals good credit and 0 equals bad credit
        self.y = dataframe["credit"].replace(2,0)
        self.sensitive_attributes = dataframe["sex"]
        #(rows,columns)
        self.shape = dataframe.shape
        self.attributes = self.X.columns
        #Copy of pd.get_dummies(self.X)?
        self.X_numerical = pd.get_dummies(self.X)
        self.names = self.X_numerical.columns
        
        #4. Create numerical labels and scale data. Format: Numerical-Binary
        if scale:
            #Is a copy necessary? 
            X_numerical = self.X_numerical.copy()
            sc = StandardScaler()
            X_preprocessed = sc.fit_transform(X_numerical)
            X_preprocessed = pd.DataFrame(X_preprocessed, columns = X_numerical.columns)
            self.X_preprocessed = X_preprocessed
            
    #5. Split data into training and testing data.
    def split_data(self):
        X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
                                                            self.X_preprocessed,
                                                            self.y,
                                                            self.sensitive_attributes,
                                                            test_size = 0.2,
                                                            random_state = 0,
                                                            stratify = self.y)
        #Reset index, why do we not reset y_train and y_test as well?
        X_train = X_train.reset_index(drop = True)
        X_test = X_test.reset_index(drop = True)
        S_train = S_train.reset_index(drop = True)
        S_test = S_test.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)
        return X_train, X_test, y_train, y_test, S_train, S_test