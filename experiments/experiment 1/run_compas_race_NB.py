import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from fair.data.objects.compas_obj import CompasData
from fair.algorithms.preprocessing.massaging import Massaging
from fair.algorithms.preprocessing.preferential_sampling import PreferentialSampling
from fair.algorithms.preprocessing.reweighing import Reweighing
from fair.algorithms.inprocessing.disparate_impact_score import DisparateImpactScore
from fair.algorithms.inprocessing.grid_search import Grid_Search
from fair.algorithms.preprocessing.disparate_impact_remover import DisparateImpactRemover
from fair.algorithms.inprocessing.exponentiated_gradient import Exponentiated_Gradient
from fair.algorithms.postprocessing.equality_of_opportunity import EqualityOfOpportunity
from fair.algorithms.preprocessing.correlation_remover import CorrelationRemover
from fair.metrics import MetricFrame, selection_rate, count, false_negative_rate, false_positive_rate, true_positive_rate, true_negative_rate
from fair.metrics._disparities import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio
from sklearn import metrics as skm

#To obtain results for sensitive attribute race change sensitive parameter to race and
#change sens_group_name[0] to index 1

df_obj = CompasData()
results = pd.DataFrame(columns = ["algorithm", "run_id", "accuracy","TPR", "TNR", "FPR", "FNR"])
results_by_group = pd.DataFrame(columns = ["algorithm", "run_id", "Race", "Count", "Selection_rate", "Accuracy", "TPR", "TNR", "FPR", "FNR"])    
disparities = pd.DataFrame(columns = ["algo", "DP-Diff", "DP-Ratio", "EO-Diff", "EO-Ratio"])
algo_names = ["Unmitigated-NB", "Massaging", "Pref. Sampling", "Reweighing", "Corr. Remover",
              "DI Remover", "Grid Search DP", "Grid Search EO", "Exp. Grad. DP", "Exp. Grad. EO", "DI Score", "Equal. Oppor. DP", "Equal. Oppor. EO"]
algo_names_dup = ["Unmitigated-LR", "Unmitigated-LR", "Massaging", "Massaging", "Pref. Sampling", "Pref. Sampling", "Reweighing","Reweighing", "Corr. Remover","Corr. Remover",
              "DI Remover","DI Remover", "Grid Search DP","Grid Search DP", "Grid Search EO", "Grid Search EO", "Exp. Grad. DP", "Exp. Grad. DP","Exp. Grad. EO", 
              "Exp. Grad. EO", "DI Score", "DI Score", "Equal. Oppor. DP", "Equal. Oppor. DP", "Equal. Oppor. EO", "Equal. Oppor. EO"]
mean_results_overall = pd.DataFrame(algo_names)
mean_results_by_group = pd.DataFrame(algo_names_dup)
mean_results_overall.columns = mean_results_by_group.columns = ['algorithm']

# dp_ratio_baseline = demographic_parity_difference(df_obj.y, df_obj.y, sensitive_features = df_obj.sensitive_attributes["race"])
# dp_diff_baseline = demographic_parity_ratio(df_obj.y, df_obj.y, sensitive_features = df_obj.sensitive_attributes["race"])
# baseline = pd.DataFrame(columns = ["accuracy", "DP-diff", "DP-ratio"])
# baseline = baseline.append({"accuracy":1, "DP-diff": dp_diff_baseline, "DP-ratio": dp_ratio_baseline}, ignore_index=True)

for i in range(10):
    #parameter sensitive = 'sex' you can choose between sex and race
    X_train, X_test, y_train, y_test, S_train, S_test, S_train_num, S_test_num = df_obj.train_test_split(train_size = 0.7, sensitive = 'race')
    
    
    unmitigated_predictor = LogisticRegression(solver = 'liblinear', fit_intercept =True)
    unmitigated_predictor.fit(X_train, y_train)
    y_pred_unmitigated = unmitigated_predictor.predict(X_test)
    results_unmitigated = pd.DataFrame({"unmitigated": y_pred_unmitigated})
    
    unmitigate_predictor_NB = GaussianNB()
    unmitigate_predictor_NB.fit(X_train, y_train)
    y_pred_NB = unmitigate_predictor_NB.predict(X_test)
    
    #1. Run Preprocessing algorithms
    #sensitive_group_name[0] for race and sensitive_group_name[1] for sex
    mssg_obj = Massaging(X_train, y_train, S_train, sens_group_name = df_obj.sens_group_name[1], 
                          non_sens_group_name = df_obj.non_sens_group_name[1], clf=GaussianNB())
    mssg_obj.fit()
    y_pred_mssg = mssg_obj.predict(X_test)
    print("Massaging on", df_obj.data_name, "done.")
    results_mssg = pd.DataFrame({"Massaging": y_pred_mssg})
    
    
    samp_obj = PreferentialSampling(X_train, y_train, S_train, 
                                    sens_group_name = df_obj.sens_group_name[1], 
                                    non_sens_group_name = df_obj.non_sens_group_name[1], clf=GaussianNB())
    samp_obj.fit()
    y_pred_samp = samp_obj.predict(X_test)
    print("Preferential Sampling on", df_obj.data_name, "done.")
    results_samp = pd.DataFrame({"Pref. Sampling": y_pred_samp})
    
    
    rewei_obj = Reweighing(X_train, y_train, S_train, sens_group_name = df_obj.sens_group_name[1],
                            non_sens_group_name = df_obj.non_sens_group_name[1], clf=GaussianNB())
    rewei_obj.fit()
    y_pred_rewei = rewei_obj.predict(X_test)
    print("Reweighing on", df_obj.data_name, "done.")
    results_rewei = pd.DataFrame({"Reweighing": y_pred_rewei})
    
    corr_obj = CorrelationRemover(sensitive_feature_ids = df_obj.sens_attr_names)
    corr_obj.fit(X_train, y_train)
    X_corr, X_sensitive = corr_obj.transform(X_train)
    corr_predictor = GaussianNB()
    corr_predictor.fit(X_corr, y_train)
    y_pred_corr = corr_predictor.predict(X_test.drop(df_obj.sens_attr_names, axis =1))
    print("Correlation Remover on", df_obj.data_name, "done.")
    
    #sens_attr_names[1] = race, sens_attr_names[0] = sex
    dispRemover_obj = DisparateImpactRemover(X_train, X_test, y_train, y_test, 
                                              ['race'], clf=GaussianNB())
    dispRemover_obj.fit()
    y_pred_dispRemover = dispRemover_obj.predict()
    print("Disparate Impact Remover on", df_obj.data_name, "done.")
    results_dispRemover = pd.DataFrame({"Disp. Impact Remover": y_pred_dispRemover})
    
    exp_gradient_obj = Exponentiated_Gradient(df_obj.sens_attr_names[1], constraints = "demographic parity", estimator=GaussianNB())
    exp_gradient_obj.fit(X_train, y_train, S_train)
    y_pred_exp_gradient = exp_gradient_obj.predict(X_test)
    print("Exponentiated Gradient DP on", df_obj.data_name, "done.")
    results_exp = pd.DataFrame({"Exp. Gradient": y_pred_exp_gradient})
    
    grid_search_obj = Grid_Search(df_obj.sens_attr_names[1], constraints = "demographic parity", estimator=GaussianNB())
    grid_search_obj.fit(X_train, y_train, S_train)
    y_pred_grid = grid_search_obj.predict(X_test)
    print("Grid Search DP on", df_obj.data_name, "done.")
    results_grid = pd.DataFrame({"Grid Search": y_pred_grid})
    
    exp_gradient_obj_2 = Exponentiated_Gradient(df_obj.sens_attr_names[1], constraints = "equalized odds", estimator=GaussianNB())
    exp_gradient_obj_2.fit(X_train, y_train, S_train)
    y_pred_exp_gradient_2 = exp_gradient_obj_2.predict(X_test)
    print("Exponentiated Gradient EO on", df_obj.data_name, "done.")
    results_exp_2 = pd.DataFrame({"Exp. Gradient": y_pred_exp_gradient_2})
    
    grid_search_obj_2 = Grid_Search(df_obj.sens_attr_names[1], constraints = "equalized odds", estimator=GaussianNB())
    grid_search_obj_2.fit(X_train, y_train, S_train)
    y_pred_grid_2 = grid_search_obj_2.predict(X_test)
    print("Grid Search EO on", df_obj.data_name, "done.")
    results_grid_2 = pd.DataFrame({"Grid Search": y_pred_grid_2})
    
    disp_imp_score_obj = DisparateImpactScore()
    w = disp_imp_score_obj.fit(dataset = df_obj, clstype = 'lin-svm', sensitive = df_obj.sens_attr_names[1])
    y_pred_zafar = disp_imp_score_obj.predict(X_test)
    print("Disparate Impact Score on", df_obj.data_name, "done.")
    results_zafar = pd.DataFrame({"Zafar": y_pred_zafar})
    
    eq_of_oppor_obj = EqualityOfOpportunity(estimator=GaussianNB())
    eq_of_oppor_obj.fit(X_train, y_train, S_train)
    y_pred_eq_opp = eq_of_oppor_obj.predict(X_test, S_test)
    print("Equality of Opportunity DP on", df_obj.data_name, "done.")
    results_equality = pd.DataFrame({"Equality": y_pred_eq_opp})
    
    eq_of_oppor_obj_2 = EqualityOfOpportunity(constraints='equalized_odds',estimator=GaussianNB())
    eq_of_oppor_obj_2.fit(X_train, y_train, S_train)
    y_pred_eq_op_2 = eq_of_oppor_obj_2.predict(X_test, S_test)
    print("Equality of Opportunity EO on", df_obj.data_name, "done.")
    results_equality_2 = pd.DataFrame({"Equality": y_pred_eq_opp})
    
    #Evaluation + Results
    n = 0
    for l in [y_pred_NB, y_pred_mssg, y_pred_samp, y_pred_rewei, y_pred_corr,
              y_pred_dispRemover, y_pred_grid, y_pred_grid_2, y_pred_exp_gradient,
              y_pred_exp_gradient_2, y_pred_zafar, y_pred_eq_opp, y_pred_eq_op_2]:
        
        demo = demographic_parity_difference(y_test, l, sensitive_features = S_test)
        demo1 = demographic_parity_ratio(y_test, l, sensitive_features= S_test)
        eq = equalized_odds_difference(y_test, l, sensitive_features = S_test)
        eq1 = equalized_odds_ratio(y_test, l, sensitive_features = S_test)
        
        disparities = disparities.append({"algo": algo_names[n], "DP-Diff": demo, "DP-Ratio": demo1, 
                                          "EO-Diff": eq, "EO-Ratio": eq1}, 
                                          ignore_index=True)
    
        metric_frame = MetricFrame(metric = {"Count": count, "Accuracy": skm.accuracy_score,
                                            "Selection_rate": selection_rate,
                                            "TPR": true_positive_rate,
                                            "TNR": true_negative_rate,
                                            "FPR": false_positive_rate,
                                            "FNR": false_negative_rate},
                                    sensitive_features = S_test,
                                    y_true = y_test,
                                    y_pred = l)
    
        results = results.append({'algorithm': algo_names[n], 'run_id': i, 
                                          'accuracy': metric_frame.overall["Accuracy"],
                                          'TPR' : metric_frame.overall["TPR"],
                                          'TNR': metric_frame.overall["TNR"],
                                          'FPR': metric_frame.overall["FPR"],
                                          'FNR': metric_frame.overall["FNR"]}, 
                                          ignore_index=True)
        k = 0
        print(algo_names[n],metric_frame.by_group['Selection_rate'])
        for p in [df_obj.sens_group_name[1], df_obj.non_sens_group_name[1]]:
            results_by_group = results_by_group.append({'algorithm':algo_names[n], 'run_id': i,
                                                        'Race': p, 'Count': metric_frame.by_group["Count"][k],
                                                        'Accuracy': metric_frame.by_group["Accuracy"][k],
                                                        'Selection_rate': metric_frame.by_group["Selection_rate"][k],
                                                        'TPR': metric_frame.by_group["TPR"][k],
                                                        'TNR': metric_frame.by_group["TNR"][k],
                                                        'FPR': metric_frame.by_group["FPR"][k],
                                                        'FNR': metric_frame.by_group["FNR"][k]},
                                                      ignore_index=True)
            k = k + 1
        n=n+1
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ITERATION", i, "DONE!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
results_overall = pd.concat([results, disparities.drop("algo", axis =1)], axis=1)
results_overall.sort_values(by = ['algorithm', 'run_id'], inplace = True)
results_by_group.sort_values(by = ['algorithm', 'run_id'], inplace = True)

df_means_overall = df_means_by_group = pd.DataFrame()
for y in algo_names:
    means_overall = results_overall[results_overall["algorithm"] == y].drop(labels='run_id', axis = 1).mean().to_frame().T
    df_means_overall = df_means_overall.append(means_overall)
    means_by_group = results_by_group[(results_by_group["algorithm"] == y) & (results_by_group["Race"] == df_obj.sens_group_name[1])].drop(labels='run_id', axis=1).mean().to_frame().T
    df_means_by_group = df_means_by_group.append(means_by_group)
    means_by_group = results_by_group[(results_by_group["algorithm"] == y) & (results_by_group["Race"] == df_obj.non_sens_group_name[1])].drop(labels='run_id', axis=1).mean().to_frame().T
    df_means_by_group = df_means_by_group.append(means_by_group)
    
df_means_overall.reset_index(drop=True, inplace =True)
df_means_by_group.reset_index(drop=True, inplace=True)
mean_results_overall = mean_results_overall.join(df_means_overall)
df_race = results_by_group["Race"].reset_index(drop=True).head(20)
mean_results_by_group = mean_results_by_group.join([df_race, df_means_by_group])

results_overall.to_csv('experiments/results/Experiment 1/compas_race-NB-results_overall.csv')
results_by_group.to_csv('experiments/results/Experiment 1/compas_race-NB-results_by_group.csv')
mean_results_by_group.to_csv('experiments/results/Experiment 1/compas_race-NB-means_by_group.csv')
mean_results_overall.to_csv('experiments/results/Experiment 1/compas_race-NB-means_overall.csv')
# baseline.to_csv('experiments/results/Experiment 1/compas_sex-baseline.csv')

