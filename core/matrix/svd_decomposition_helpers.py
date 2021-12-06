import numpy as np
import os
import pandas as pd


def fix_scipy_svds(U, sigmas, V_T):
    r"""
    scipy.sparse.linalg.svds orders the singular values in increasing order.
    This function flips this order.

    Parameters
    ___________
    U, sigmas,V_T

    Returns
    -------
    U_nwq, sigmas_new, V_T_new
    ordered in decreasing singular values
    """
    sv_reordering = np.argsort(-sigmas)

    U_new = U[:, sv_reordering]
    sigmas_new = sigmas[sv_reordering]
    V_T_new = np.fliplr(V_T)

    return U_new, sigmas_new, V_T_new 


def serialize_SVD(U, sigmas, V_T, file_names):
    """
    function to serialize SVD output
    """
    matrices = [U,sigmas,V_T]
    for i, mat in enumerate(matrices):
        output_file_name = file_names[i]
        output_full_path = os.path.join("output", output_file_name)
        mat.dump(output_full_path, protocol=4)
        
        # load in the serialized df to make sure it matches original df
        reloaded = np.load(output_full_path, allow_pickle=True)
        assert np.array_equal(reloaded,mat)


def compute_truncated_svd_recon_err(U, sigma, V_T, k):
    """
    Function that takes the svd of a matrix and a scalar k,
    truncates the svd and returns the reconstruction error
    """
    U_k = U[:,:k]
    sigma_k = np.diag(sigma)[:k,:k]
    V_T_k = V_T[:k,:]
    
    X_k = U_k @ sigma_k @ V_T_k
    recon_error = np.sqrt((sigma**2).sum() - (sigma[:k]**2).sum())

    return recon_error


def svd_k_search(input_U, input_sigmas, input_V_T, k_vals):
    """
    function for searching over different values of k
    """
    results = []
    for kval in k_vals:
        re_err = compute_truncated_svd_recon_err(input_U, input_sigmas,
                                                 input_V_T, kval)
        entry = [kval, re_err]
        results.append(entry)
    results_df = pd.DataFrame(results, columns=['k', 'Reconstruction Error'])
    return results_df
