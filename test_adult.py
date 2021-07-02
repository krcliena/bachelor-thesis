from fair.data.objects.adult_obj import AdultData
from fair.algorithms.preprocessing.massaging import Massaging
from fair.algorithms.preprocessing.preferential_sampling import PreferentialSampling
from fair.algorithms.inprocessing.zafar_algorithm import ZafarAlgorithm
from fair.algorithms.preprocessing.reweighing import Reweighing

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

algo_zafar = ZafarAlgorithm()
w = algo_zafar.fit(dataset = ad, clstype = 'log-reg')
y_pred_zafar = algo_zafar.predict(ad.X_test)