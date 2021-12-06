import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.decomposition import NMF
from time import time


def nmf_k_helper(input_matrix, kval,write_model_to_file=False):
    """
    helper function for nmf_k search
    """
    t0=time()
    nmf_model = NMF(n_components=kval, init='nndsvd',
                    max_iter=100000,random_state=1)
    W = nmf_model.fit_transform(input_matrix)
    H = nmf_model.components_
    time_elapsed = time() - t0

    entry = [kval, nmf_model.reconstruction_err_,
                nmf_model.n_iter_, time_elapsed]
    
    return entry


def serialize_NMF(W, H, file_name):
    """
    function to serialize NMF output
    """
    matrices = [W,H]
    holder = ['W','H']

    for i, mat in enumerate(matrices):
        output_file_name = f'{file_name}{holder[i]}.pkl'
        output_full_path = os.path.join("output", output_file_name)
        mat.dump(output_full_path, protocol=4)
        
        # load in the serialized df to make sure it matches original df
        reloaded = np.load(output_full_path, allow_pickle=True)
        assert np.array_equal(reloaded,mat)


def nmf_k_search(input_matrix, k_vals, serialize=False):
    """
    function for searching over different values of k
    """
    
    results = []
    for kval in k_vals:
        print(f"Now fitting NMF for k ={kval}...")
        
        t0=time()
        nmf_model = NMF(n_components=kval, init='nndsvd',
                        max_iter=1000,random_state=1)
        W = nmf_model.fit_transform(input_matrix)
        time_elapsed = time() - t0
        H = nmf_model.components_

        if serialize == True:
            filename = f'NMF_{kval}k'
            serialize_NMF(W, H, filename)
        
        entry = [kval, nmf_model.reconstruction_err_,
                 nmf_model.n_iter_, time_elapsed]
        
        print(entry)
        
        results.append(entry)
        
    results_df = pd.DataFrame(results, columns=['k', 'Reconstruction Error',
                                                'Iterations to convergence',
                                                'Time to converge (secs)'])
    
    return results_df
