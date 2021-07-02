# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:20:19 2021

@author: Karine Louis
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CompasData():
    
    #Export preprocessed data? 
    def __init__(self, scale = True):
        
        #1. Define class attributes belonging to Adult Dataset
        self.df = pd.read_csv(r'C:\Users\Rafik\Desktop\BA\fair-main\fair\data\raw\compas-scores-two-years.csv')
        self.data_name = 'propublica-recidivism'
        self.sensitive_attributes_names = ['sex','race']
        #Should I put them in the same class?
        self.priv_class_names = ['Male', 'Caucasian']
        #Here we have a punitive example so class = 1 is a considered bad 
        self.pos_class = 1
        #Is this copy necessary, if I were to use self.df.replace would it be a reference to the obj?
        dataframe = self.df.copy()
        #dataframe = dataframe.replace({"Caucasian": "Other", "Hispanic": "Other"})
        
        
        features_to_drop = ["id", "name", "first", "last", "compas_screening_date", 
                    "dob", "c_jail_in", "c_jail_out", "c_case_number", 
                    "c_offense_date", "c_arrest_date", "c_days_from_compas",
                    "r_case_number", "r_charge_degree", "r_days_from_arrest", 
                    "r_offense_date","r_charge_desc", "r_jail_in","r_jail_out", 
                    "violent_recid", "is_violent_recid", "vr_case_number", 
                    "vr_charge_degree", "vr_offense_date", "vr_charge_desc", 
                    "type_of_assessment", "screening_date", 
                    "v_type_of_assessment", "v_decile_score", "v_score_text", 
                    "v_screening_date", "in_custody", "out_custody", 
                    "priors_count", "start", "end", "event"]
        
        
        dataframe = dataframe.drop(features_to_drop, axis = 1).dropna()
        #3. Define attributes, sensitive and target data
        self.X = dataframe.drop('two_year_recid', axis = 1)
        self.target = dataframe["two_year_recid"]
        self.sensitive_attributes = dataframe[self.sensitive_attributes_names]
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