# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:58:38 2021

@author: Rafik
"""
from sklearn.datasets import fetch_openml

#data_home????
def fetch_german_credit(*, cache=True, data_home=None,
                 as_frame=False, return_X_y=False):

    return fetch_openml(
        data_id=31,
        data_home=data_home,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
    )
