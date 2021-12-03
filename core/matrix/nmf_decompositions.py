import os
import scipy.sparse as sp
import pandas as pd
from sklearn.decomposition import NMF
from time import time



def nmf_k_helper(input_matrix, kval,write_model_to_file=False):
    """
    helper function for nmf_k search
    """
    t0=
    nmf_model = NMF(n_components=kval, init='nndsvd',
                    max_iter=100000,random_state=1)
    W = nmf_model.fit_transform(input_matrix)
    H = nmf_model.components_
    time_elapsed = time() - t0)

    entry = [kval, nmf_model.reconstruction_err_,
                nmf_model.n_iter_, time_elapsed]
    
    return entry