import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AdultData():
    
    #Export preprocessed data? 
    def __init__(self, scale = True):
        
        #1. Define class attributes belonging to Adult Dataset
        self.df = pd.read_csv(r'C:\Users\Rafik\Desktop\BA\fair-main\fair\data\raw\adult.csv')
        self.data_name = 'adult'
        self.sensitive_attributes_names = ['Sex']
        self.priv_class_name = 'Male'
        self.pos_class = '>50K'
        #Is this copy necessary, if I were to use self.df.replace would it be a reference to the obj?
        dataframe = self.df.copy()
        
        #Individuals with "workclass" = "Never-worked" have an occupation of '?',
        #replace these "?" with "No-occupation". Drop rest. 
        indices = dataframe[dataframe["workclass"] == " Never-worked"].index
        for index in indices:
            dataframe.loc[index, 'occupation'] = ' No-occupation'
        
        #2. Drop missing data, missing label indication: '?'
        df_dropped = dataframe.replace(' ?', np.nan).dropna()
        
        #3. Define attributes, sensitive and target data
        self.X = df_dropped.drop('class', axis = 1)
        self.target = df_dropped["class"]
        #Transform target class into binary numerical, '>50K' is positive class
        self.y = (df_dropped["class"] == ' >50K') * 1
        self.sensitive_attributes = df_dropped["sex"]
        #(rows,columns)
        self.shape = df_dropped.shape
        self.attributes = self.X.columns
        #Copy of pd.get_dummies(self.X)?
        self.X_numerical = pd.get_dummies(self.X)
        self.names = self.X_numerical.columns
        
        if not isinstance(self.priv_class_name, int):
            sens_numerical_arr = self.sensitive_attributes.replace({' Female': 0, ' Male' : 1})
        self.sens_numerical_arr = sens_numerical_arr
        
        #4. Create numerical labels and scale data. Format: Numerical-Binary
        if scale:
            #Is a copy necessary? 
            X_numerical = self.X_numerical.copy()
            sc = StandardScaler()
            X_preprocessed = sc.fit_transform(X_numerical)
            X_preprocessed = pd.DataFrame(X_preprocessed, columns = X_numerical.columns)
            self.X_preprocessed = X_preprocessed
            
    #5. Split data into training and testing data.
    def train_test_split(self, train_size= 0.8, val= None):
        X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
                                                            self.X_preprocessed,
                                                            self.y,
                                                            self.sens_numerical_arr,
                                                            train_size = train_size,
                                                            random_state = 0,
                                                            stratify = self.y)
        #Reset index, why do we not reset y_train and y_test as well?
        X_train = X_train.reset_index(drop = True)
        X_test = X_test.reset_index(drop = True)
        S_train = S_train.reset_index(drop = True)
        S_test = S_test.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.S_train = S_train
        self.S_test = S_test
        
        return X_train, X_test, y_train, y_test, S_train, S_test