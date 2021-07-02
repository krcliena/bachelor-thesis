# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:20:21 2021

@author: Karine Louis
"""

# =============================================================================
# This research employed a binary variable, default payment (Yes = 1, No = 0),
#as the response variable. This study reviewed the literature and used the 
#following 23 variables as explanatory variables:
#X1: Amount of the given credit (NT dollar): it includes both the individual 
#consumer credit and his/her family (supplementary) credit.
#X2: Gender (1 = male; 2 = female).
#X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
#X4: Marital status (1 = married; 2 = single; 3 = others).
#X5: Age (year).
#X6 - X11: History of past payment. We tracked the past monthly payment records 
#(from April to September, 2005) as follows: X6 = the repayment status in 
#September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the 
#repayment status in April, 2005. The measurement scale for the repayment 
#status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment 
#delay for two months; . . .; 8 = payment delay for eight months; 9 = payment 
#delay for nine months and above.
#X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement 
#in September, 2005; X13 = amount of bill statement in August, 2005; . . .; 
#X17 = amount of bill statement in April, 2005.
#X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in 
#September, 2005; #X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

# =============================================================================
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LoanDefaultsData():
    
    #Export preprocessed data? 
    def __init__(self, scale = True):
        
        #1. Define class attributes belonging to Adult Dataset
        self.df = pd.read_csv(r'C:\Users\Rafik\Desktop\BA\fair-main\fair\data\raw\loan_defaults.csv')
        self.data_name = 'loan defaults'
        self.sensitive_attributes_names = ['sex']
        self.priv_class_name = 'male'
        self.pos_class = 1
        #Is this copy necessary, if I were to use self.df.replace would it be a reference to the obj?
        dataframe = self.df.copy().drop('id', axis =1)
        #column names
        attributes = ["given-credit", "sex", "education", "marital-status", "age", 
              "payment-Sep","payment-Aug","payment-Jul", "payment-Jun",
              "payment-May","payment-Apr", "bill-statement-Sep", 
              "bill-statement-Aug", "bill-statement-Jul", "bill-statement-Jun",
              "bill-statement-May", "bill-statement-Apr", "previous-payment-Sep",
              "previous-payment-Aug", "previous-payment-Jun", "previous-payment-Jul",
              "previous-payment-May","previous-payment-Apr", "default-payment"]
        dataframe.columns = attributes
        
        #Individuals with "workclass" = "Never-worked" have an occupation of '?',
        #replace these "?" with "No-occupation". Drop rest. 
        #indices = dataframe[dataframe["workclass"] == " Never-worked"].index
        #for index in indices:
        #    dataframe.loc[index, 'occupation'] = ' No-occupation'
        
        #2. Drop missing data, missing label indication: '?'
        #df_dropped = dataframe.replace(' ?', np.nan).dropna()
        
        #3. Define attributes, sensitive and target data
        self.X = dataframe.drop('default-payment', axis = 1)
        self.target = dataframe["default-payment"]
        # #1 equals good credit and 0 equals bad credit
        self.y = self.target.copy()
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