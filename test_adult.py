from fair.data.objects.adult_obj import AdultData
from fair.algorithms.preprocessing.massaging import Massaging
from fair.algorithms.preprocessing.preferential_sampling import PreferentialSampling
from fair.algorithms.inprocessing.zafar_algorithm import ZafarAlgorithm
from fair.algorithms.preprocessing.reweighing import Reweighing
from fair.algorithms.inprocessing.grid_search import GridSearch
from fair.algorithms.inprocessing.exponentiated_gradient import ExponentiatedGradient


ad = AdultData()
X_train, X_test, y_train, y_test, A_train, A_test = ad.train_test_split(train_size = 0.7)


algo_msg = Massaging(X_train, y_train, A_train, sens_group_name = " Female", 
                     non_sens_group_name = " Male")
algo_msg.fit()
y_pred_msg = algo_msg.predict(X_test)


algo_samp = PreferentialSampling(X_train, y_train, A_train, sens_group_name = " Female",
                                     non_sens_group_name = " Male")
algo_samp.fit()
y_pred_samp = algo_samp.predict(X_test)

algo_rew = Reweighing(X_train, y_train, A_train, sens_group_name = " Female", 
                     non_sens_group_name = " Male")
W, S_C, weight = algo_rew.reweighing()
algo_rew.fit()
y_pred_rew = algo_rew.predict(X_test)

#algo_zafar = ZafarAlgorithm()
#w = algo_zafar.fit(dataset = ad, clstype = 'log-reg')
#y_pred_zafar = algo_zafar.predict(ad.X_test)

algo_gridSearch = GridSearch()
algo_gridSearch.fit(X_train, y_train, A_train)
y_pred_grid = algo_gridSearch.predict(X_test)

algo_exponGradient = ExponentiatedGradient()
algo_exponGradient.fit(X_train, y_train, A_train)
y_pred_exponen = algo_exponGradient.predict(X_test)
