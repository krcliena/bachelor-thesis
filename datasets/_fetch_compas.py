# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:23:36 2021

@author: Karine Louis
"""


from sklearn.datasets import fetch_openml

#data_home???
def fetch_compas(*, cache=True, data_home=None,
                 as_frame=False, return_X_y=False):

    return fetch_openml(data_id =42193, cache = cache, data_home = data_home,
                        as_frame = as_frame, return_X_y= return_X_y)
