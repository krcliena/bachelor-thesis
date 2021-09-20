import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GermanData():
"""German Statlog Data. 
   See :file:`fairML/data/README.md` for raw dataset.
   This dataset includes 1000 instances and 20 attributes.
   Sensitive attributes are Age and Gender.
   Privileged Groups are older than 25 and Male.
   Predicts whether an individual is a good or bad credit risk.
   
   Data specific pre-processing is done.
   """
    def __init__(self, scale = True):
        
        #1. Define class attributes belonging to Adult Dataset
        self.df = pd.read_csv(r'C:\Users\new\Desktop\Karine\BA\fair-main\fair\data\raw\german_credit.csv')
        self.data_name = 'german'
        self.sens_attr_names = ['sex', 'age']
        self.sens_group_name = ['female', 'young']
        self.non_sens_group_name = ['male', 'aged']
        self.df = self.df.replace({'female': 0, 'male': 1})
        dataframe = self.df.copy()
        
        #Individuals with "workclass" = "Never-worked" have an occupation of '?',
        #replace these "?" with "No-occupation". Drop rest. 
        #indices = dataframe[dataframe["workclass"] == " Never-worked"].index
        #for index in indices:
        #    dataframe.loc[index, 'occupation'] = ' No-occupation'
        
        #2. Drop missing data, missing label indication: '?'
        #df_dropped = dataframe.replace(' ?', np.nan).dropna()
        dataframe.loc[dataframe['age'] >= 25, 'age'] = 'aged'
        dataframe.loc[dataframe['age'] != 'aged', 'age'] = 'young'
        
        self.sensitive_attributes = dataframe[self.sens_attr_names].replace({1:'male', 0: 'female'})
        self.sens_attributes_num = dataframe[self.sens_attr_names].replace({'aged': 1, 'young': 0})
        
        #3. Define attributes, sensitive and target data
        self.X = dataframe.drop('credit', axis = 1).replace({'aged': 1, 'young': 0})
        #self.target = dataframe["credit"]
        # #1 equals good credit and 0 equals bad credit
        self.y = dataframe["credit"].replace(2,0)
        #(rows,columns)
        #self.shape = dataframe.shape
        self.attributes = self.X.columns
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
    def train_test_split(self, train_size = 0.8, val = None, sensitive = 'age'):
        """Serves as a wrapper around Scikit-learn's train_test_split function. Returns the train and test splits on the data.
        'sensitive': Choose which sensitive attribute to split on, allowed inputs: 'sex' and 'age'
        Returns
        -------
        pd.DataFrame
            Train and test splits on X, y, sensitive attributes S, sensitive attributes numerical S_num, other sensitive attribute S_oth
        """
        
        X_train, X_test, y_train, y_test, S_train, S_test, S_train_num, S_test_num, S_train_oth, S_test_oth = train_test_split(
                                                                                                                self.X_preprocessed,
                                                                                                                self.y,
                                                                                                                self.sensitive_attributes[sensitive],
                                                                                                                self.sens_attributes_num[sensitive],
                                                                                                                self.sensitive_attributes.drop(columns = sensitive),
                                                                                                                train_size = train_size,
                                                                                                                stratify = self.y)
        #Reset index, why do we not reset y_train and y_test as well?
        X_train = X_train.reset_index(drop = True)
        X_test = X_test.reset_index(drop = True)
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
