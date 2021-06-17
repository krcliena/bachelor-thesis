# -*- coding: utf-8 -*-
"""
@author: Karine Louis
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as skm
from fair.datasets import _fetch_adult
from fair.preprocessing import CorrelationRemover
from fair.metrics import MetricFrame, selection_rate, count

df_adult = _fetch_adult.fetch_adult(as_frame = True)
X_raw = df_adult.data
y = (df_adult.target == '>50K') * 1
#sensitive feature = sex
sens = X_raw["sex"]
X = pd.get_dummies(X_raw)
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)

#correlationRemover alpha = 1
corr = CorrelationRemover(sensitive_feature_ids = ["sex_Male", "sex_Female"])
hi = corr.fit(X, y)
X_corr, X_sensitive = corr.transform(X)

#correlationRemover alpha = 0.5
corr2 = CorrelationRemover(sensitive_feature_ids = ["sex_Male", "sex_Female"], 
                           alpha = 0.2)
corr2.fit(X, y)
X_corr_2, X_sensitive_2 = corr2.transform(X)

#scaled transformed Data
X_corr_scaled = sc.fit_transform(X_corr)
X_corr2_scaled = sc.fit_transform(X_corr_2)

#name columns in X_corr, are the columns correct?
columns = X.columns.drop(["sex_Female", "sex_Male"])
X_corr_scaled = pd.DataFrame(X_corr_scaled, columns = columns)
X_corr2_scaled = pd.DataFrame(X_corr2_scaled, columns = columns)

#Training unmitigated and not preprocessed Model, A_train and A_test keep memory
#of sensitive attributes 
X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X,
                                                                    y, sens, 
                                                                    test_size = 0.2,
                                                                    random_state = 0,
                                                                    stratify = y)
#X_train = X_train.reset_index(drop = True)
#X_test = X_test.reset_index(drop = True)
#A_train = A_train.reset_index(drop = True)
#A_test = A_test.reset_index(drop = True)
unmitigated_predictor = LogisticRegression(solver = 'liblinear', 
                                           fit_intercept = True)
unmitigated_predictor.fit(X_train, Y_train)
y_pred = unmitigated_predictor.predict(X_test)

#Training preprocessed Model
X_train_pre, X_test_pre, Y_train_pre, Y_test_pre, A_train_pre, A_test_pre = train_test_split(
X_corr, y, sens, test_size = 0.2, random_state = 0, stratify = y)

#X_train_pre = X_train_pre.reset_index(drop = True)
#X_test_pre =  X_test_pre.reset_index(drop = True)
#A_train_pre = A_train_pre.reset_index(drop = True)
#A_test_pre = A_test_pre.reset_index(drop = True)
preprocessed_predictor = LogisticRegression(solver = 'liblinear', 
                                            fit_intercept = True)
preprocessed_predictor.fit(X_train_pre, Y_train_pre)
y_pred_pre = preprocessed_predictor.predict(X_test_pre)

#assess predictors' fairness using MetricFrame from Fairlearn
metric_frame = MetricFrame(metric = {"Accuracy": skm.accuracy_score,
                                     "Selection Rate": selection_rate,
                                     "Count": count},
                           sensitive_features = A_test,
                           y_true = Y_test,
                           y_pred = y_pred)
print("Metric Frame for Unmitigated Model:")
print(metric_frame.overall)
print(metric_frame.by_group)
#plot bars of metrics
metric_frame.by_group.plot.bar(
        subplots=True, layout=[3, 1], legend=False, figsize=[12, 8],
        title='Accuracy and selection rate by group')
# Looking at the disparity in accuracy, we see that males have an error
# about three times greater than the females.
# More interesting is the disparity in opportunity - males are offered loans at
# three times the rate of females.

metric_frame_pre = MetricFrame(metric = {"Accuracy": skm.accuracy_score,
                                         "selection_rate": selection_rate,
                                         "Count": count},
                               sensitive_features = A_test_pre,
                               y_true = Y_test_pre,
                               y_pred = y_pred_pre)

print("Metric Frame for Preprocessed Model:")
print(metric_frame_pre.overall)
print(metric_frame_pre.by_group)