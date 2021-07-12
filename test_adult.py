from fair.data.objects.adult_obj import AdultData
from fair.algorithms.preprocessing.massaging import Massaging
from fair.algorithms.preprocessing.preferential_sampling import PreferentialSampling
from fair.algorithms.inprocessing.zafar_algorithm import ZafarAlgorithm
from fair.algorithms.preprocessing.reweighing import Reweighing
from fair.algorithms.inprocessing.grid_search import GridSearch
from fair.algorithms.preprocessing.disparate_impact_remover import DisparateImpactRemover
from fair.algorithms.inprocessing.exponentiated_gradient import ExponentiatedGradient
from fair.algorithms.postprocessing.threshhold_optimizer import EqualityOfOpportunity
from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#from fair.aif360.algorithms.preprocessing import DisparateImpactRemover
from fair.aif360.datasets import AdultDataset
#from fair.aif360.metrics import BinaryLabelDatasetMetric



ad = AdultData()
X_train, X_test, y_train, y_test, A_train, A_test = ad.train_test_split(train_size = 0.7)


#algo_msg = Massaging(X_train, y_train, A_train, sens_group_name = " Female", 
                     #non_sens_group_name = " Male")
#algo_msg.fit()
#y_pred_msg = algo_msg.predict(X_test)


#algo_samp = PreferentialSampling(X_train, y_train, A_train, sens_group_name = " Female",
#                                     non_sens_group_name = " Male")
#algo_samp.fit()
#y_pred_samp = algo_samp.predict(X_test)

#algo_rew = Reweighing(X_train, y_train, A_train, sens_group_name = " Female", 
#                    non_sens_group_name = " Male")
#W, S_C, weight = algo_rew.reweighing()
#algo_rew.fit()
#y_pred_rew = algo_rew.predict(X_test)

#algo_zafar = ZafarAlgorithm()
#w = algo_zafar.fit(dataset = ad, clstype = 'log-reg')
#y_pred_zafar = algo_zafar.predict(ad.X_test)

#algo_gridSearch = GridSearch()
#algo_gridSearch.fit(X_train, y_train, A_train)
#y_pred_grid = algo_gridSearch.predict(X_test)

#algo_exponGradient = ExponentiatedGradient()
#algo_exponGradient.fit(X_train, y_train, A_train)
#y_pred_exponen = algo_exponGradient.predict(X_test)

#What's kdd?

 
#all_data s

#adult_dataset = ad.df
#adult_dataset = pd.get_dummies(adult_dataset)
#scaler = MinMaxScaler(copy=False)

#adult_dataset = scaler.fit_transform(adult_dataset)
repairer = Repairer(X_train.to_numpy(), 104, 1.0, False)
repaired_data = repairer.repair(X_train.to_numpy().tolist())
repaired_data = pd.DataFrame(repaired_data, columns = X_train.columns)

#protected = 'sex'
#ad_new = AdultDataset(protected_attribute_names=[protected],
#    privileged_classes=[['Male']], categorical_features=[],
#    features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])
#test, train = ad_new.split([16281])
#train.features = scaler.fit_transform(train.features)
#test.features = scaler.fit_transform(test.features)
#index = train.feature_names.index(protected)

#list_adult = X_train.to_numpy().tolist()
#new_list = len(list_adult[0])

#algo_dispRemover = DisparateImpactRemover(X_train, X_test, y_train, y_test, " sex")
#algo_dispRemover.fit()
#y_pred_dispRemover = algo_dispRemover.predict()

algo_equalityOfOppor = EqualityOfOpportunity()

algo_equalityOfOppor.fit(X_train,y_train, A_train)

y_pred_equalityOfOppor = algo_equalityOfOppor.predict(X_test, A_test)
