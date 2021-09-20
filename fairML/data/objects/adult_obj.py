import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class AdultData():
"""Adult Census Income Data. 
   See :file:`fairML/data/README.md` for raw dataset.
   This dataset includes 48,842 instances and 14 attributes.
   Sensitive attributes are Race and Gender.
   Privileged Groups are White and Male.
   Predicts whether an individual will make >50K/year.
   
   Data specific pre-processing is done.
   """

    def __init__(self, scale = True):
        
        #1. Define class attributes belonging to Adult Dataset
        self.df = pd.read_csv(r'C:\Users\new\Desktop\Karine\BA\fair-main\fair\data\raw\adult.csv')
        self.data_name = 'adult'
        self.sens_attr_names = ['sex', 'race']
        self.sens_group_name = ['Female', 'Black']
        self.non_sens_group_name = ['Male', 'Other']
        self.df = self.df.replace({' Female': 0, ' Male': 1})
        self.df = self.df.replace({' Black': 'Black', ' White': 'Other', ' Other': 'Other', 
                                   ' Amer-Indian-Eskimo': 'Other', ' Asian-Pac-Islander': 'Other'})
        #self.pos_class = '>50K'
        dataframe = self.df.copy()

        #2. Drop missing data, missing label indication: '?'
        dataframe = dataframe.replace(' ?', np.nan)
        
        #3. Define attributes, sensitive and target data
        self.X = dataframe.drop(['class', 'fnlwgt', 'relationship', 'capital-gain', 'capital-loss'], axis = 1)
        self.target = dataframe["class"]
        #Transform target class into binary numerical, '>50K' is positive class
        self.y = (self.target == ' >50K') * 1
        self.sens_attributes_num = dataframe[self.sens_attr_names].replace({'Black': 0, 'Other': 1})
        self.sensitive_attributes = dataframe[self.sens_attr_names].replace({1: 'Male', 0: 'Female'})
        #(rows,columns)
        #self.shape = df_dropped.shape
        self.attributes = self.X.columns
        self.X_numerical = pd.get_dummies(self.X.replace({"Black":0, "Other":1}))
        
        #4. Create numerical labels and scale data. Format: Numerical-Binary
        if scale:
            #Is a copy necessary? 
            X_numerical = self.X_numerical.copy()
            self.sc = StandardScaler()
            X_preprocessed = self.sc.fit_transform(X_numerical)
            X_preprocessed = pd.DataFrame(X_preprocessed, columns = X_numerical.columns)
            self.X_preprocessed = X_preprocessed
        
            

    def train_test_split(self, train_size= 0.8, val= None, X = None, sensitive = 'sex'):
        """Serves as a wrapper around Scikit-learn's train_test_split function. Returns the train and test splits on the data.
        'sensitive': Choose which sensitive attribute to split on, allowed inputs: 'sex' and 'race'
        Returns
        -------
        pd.DataFrame
            Train and test splits on X, y, sensitive attributes S, sensitive attributes numerical S_num, other sensitive attribute S_oth
        """
        print("Splitting on sensitive attribute:", sensitive)
        if X is None:
            X = self.X_preprocessed
        if X is not None:
            X = self.sc.fit_transform(X)
            
        X_train, X_test, y_train, y_test, S_train, S_test, S_train_num, S_test_num, S_train_oth, S_test_oth = train_test_split(
                                                            self.X_preprocessed,
                                                            self.y,
                                                            self.sensitive_attributes[sensitive],
                                                            self.sens_attributes_num[sensitive],
                                                            self.sensitive_attributes.drop(columns = sensitive),
                                                            train_size = train_size,
                                                            stratify = self.y)

        S_train = S_train.reset_index(drop = True)
        S_test = S_test.reset_index(drop = True)
        S_train_num = S_train_num.reset_index(drop = True)
        S_test_num = S_test_num.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.S_train = S_train
        self.S_test = S_test
        self.S_train_num = S_train_num
        self.S_test_num = S_test_num
        
        return X_train, X_test, y_train, y_test, S_train, S_test, S_train_num, S_test_num, S_train_oth, S_test_oth
