import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CompasData():
    
"""ProPublica Recidivism Compas dataset. 
   See :file:`fairML/data/README.md` for raw dataset.
   This dataset includes 7214 instances and 53 attributes.
   Sensitive attributes are Race and Gender.
   Privileged Groups are Caucasian and male.
   Predicts whether defendant will recidiviate within two years.
   
   Data specific pre-processing is done.
   """
    

    def __init__(self, scale = True):
        
        #1. Define class attributes belonging to compas Dataset
        self.df = pd.read_csv(r'C:\Users\new\Desktop\Karine\BA\fair-main\fair\data\raw\compas-scores-two-years.csv')
        self.data_name = 'propublica-recidivism'
        self.sens_attr_names = ['sex','race']
        self.sens_group_name = ['Female', 'African-American']
        self.non_sens_group_name = ['Male', 'Other']
        #Here we have a punitive example so class = 1 is a considered bad 
        #self.pos_class = 1
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
        
        #Filtering done as here:
        #https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        #https://github.com/algofairness/fairness-comparison/blob/master/fairness/data/objects/PropublicaRecidivism.py
        dataframe = dataframe[(dataframe.days_b_screening_arrest <= 30) &
                              (dataframe.days_b_screening_arrest >= -30) &
                              (dataframe.is_recid != -1) &
                              (dataframe.c_charge_degree != '0') &
                              (dataframe.score_text != 'N/A')]
        #dataframe = dataframe.drop(columns = ['days_b_screening_arrest', 'is_recid',
                                              # 'decile_score', 'score_text'])
                                              
        dataframe = dataframe.drop(features_to_drop, axis = 1).dropna()
        #3. Define attributes, sensitive and target data
        self.X = dataframe.drop('two_year_recid', axis = 1)
        #self.target = dataframe["two_year_recid"]
        self.y = dataframe['two_year_recid']
        #(rows,columns)
        #self.shape = dataframe.shape
        self.attributes = self.X.columns
        #Sensitive attributes sex and race transformed into binary labels first
        self.X_numerical = self.X.replace({'Male': 1, 'Female': 0, 'Caucasian': 1, 'Other': 1, 
                                           'African-American': 0, 'Hispanic': 1, 
                                           'Asian': 1, 'Native American': 1})
        #Copy of pd.get_dummies(self.X)?
        self.X_numerical = pd.get_dummies(self.X_numerical)
        #self.names = self.X_numerical.columns
        self.sensitive_attributes = dataframe[self.sens_attr_names].replace({"Caucasian": "Other", 
                                                                             "Native American": "Other", 
                                                                             "Hispanic":"Other",
                                                                             "Asian": "Other"})
        self.sens_attributes_num = self.sensitive_attributes.replace({"African-American": 0, "Other": 1, "Male": 1, "Female": 0})

        #4. Create numerical labels and scale data. Format: Numerical-Binary
        if scale:
            #Is a copy necessary? 
            X_numerical = self.X_numerical.copy()
            sc = StandardScaler()
            X_preprocessed = sc.fit_transform(X_numerical)
            X_preprocessed = pd.DataFrame(X_preprocessed, columns = X_numerical.columns)
            self.X_preprocessed = X_preprocessed
            
    #5. Split data into training and testing data.
    def train_test_split(self, train_size = 0.8, val = None, sensitive = 'sex'):
    """Serves as a wrapper around Scikit-learn's train_test_split function. Returns the train and test splits on the data.
        'sensitive': Choose which sensitive attribute to split on, allowed inputs: 'sex' and 'race'
        Returns
        -------
        pd.DataFrame
            Train and test splits on X, y, sensitive attributes S, sensitive attributes numerical S_num, other sensitive attribute S_oth
        """
        print("splitting on", sensitive)
        X_train, X_test, y_train, y_test, S_train, S_test, S_train_num, S_test_num = train_test_split(self.X_preprocessed,
                                                                                                      self.y,
                                                                                                      self.sensitive_attributes[sensitive],
                                                                                                      self.sens_attributes_num[sensitive],
                                                                                                      train_size = train_size,
                                                                                                      stratify = self.y)
        #Reset index, why do we not reset y_train and y_test as well?
        X_train = X_train.reset_index(drop = True)
        X_test = X_test.reset_index(drop = True)
        S_train = S_train.reset_index(drop = True)
        S_test = S_test.reset_index(drop = True)
        S_train_num = S_train_num.reset_index(drop = True)
        S_test_num = S_test_num.reset_index(drop=True)
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
