import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RicciData():
"""Ricci v. DeStefano Data. 
   See :file:`fairML/data/README.md` for raw dataset.
   This dataset includes 118 instances and 5 attributes.
   Sensitive attribute is Race.
   Privileged Groups are Other.
   Predicts whether an individual is likely to receive a promotion.
   
   Data specific pre-processing is done.
   """
    
    #Export preprocessed data? 
    def __init__(self, scale = True):
        
        #1. Define class attributes belonging to Adult Dataset
        self.df = pd.read_csv(r'C:\Users\new\Desktop\Karine\BA\fair-main\fair\data\raw\ricci.csv')
        self.data_name = 'ricci'
        self.sens_attr_names = ['Race']
        self.sens_group_name = 'B'
        self.non_sens_group_name = 'O'
        #self.pos_class = 'Promotion'
        dataframe = self.df.copy()
        
        #3. Define attributes, sensitive and target data
        self.X = dataframe.drop('Promotion', axis = 1)
        self.target = dataframe["Promotion"]
        self.y = (dataframe["Promotion"] == 'Promotion') * 1
        self.sensitive_attributes = dataframe["Race"]
        self.sensitive_attributes_binary = dataframe["Race"].replace({"W": "O", "H": "O"})
        self.sens_attributes_num = self.sensitive_attributes_binary.replace({"O":1, "B": 0})
        #(rows,columns)
        #self.shape = dataframe.shape
        self.attributes = self.X.columns
        #Copy of pd.get_dummies(self.X)?
        self.X_numerical = pd.get_dummies(self.X.replace({"H":1, "W":1, "B":0}))
        self.names = self.X_numerical.columns
        
        #4. Create numerical labels and scale data. Format: Numerical-Binary
        if scale:
            X_numerical = self.X_numerical.copy()
            sc = StandardScaler()
            X_preprocessed = sc.fit_transform(X_numerical)
            X_preprocessed = pd.DataFrame(X_preprocessed, columns = X_numerical.columns)
            self.X_preprocessed = X_preprocessed
            
    #5. Split data into training and testing data.
    def train_test_split(self, train_size = 0.8, val = None, sensitive = None):
     """Serves as a wrapper around Scikit-learn's train_test_split function. Returns the train and test splits on the data.
        
        Parameters
        ----------
        'sensitive': Choose which sensitive attribute to split on, since only one exits, default is None.
        Returns
        -------
        pd.DataFrame
            Train and test splits on X, y, sensitive attributes S, sensitive attributes numerical S_num
        """
        X_train, X_test, y_train, y_test, S_train, S_test, S_train_num, S_test_num = train_test_split(
                                                            self.X_preprocessed,
                                                            self.y,
                                                            self.sensitive_attributes_binary,
                                                            self.sens_attributes_num,
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
        
        return X_train, X_test, y_train, y_test, S_train, S_test, S_train_num, S_test_num
