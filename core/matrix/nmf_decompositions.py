import os
from time import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


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


def compute_nmf(k, A):
    """"
    A simple wrapper function for sklearn NMF
    Instantiate NMF then factorize A into W and H

    Input:
        A: numpy.ndarray, matrix to factor
        k: (int)

    Returns:
        nmf_model: NMF model instance
        W: numpy.ndarray
        H: numpy.ndarray
    """
    nmf_model = NMF(n_components=k,
                    init='nndsvd',
                    max_iter=1000,
                    random_state=1)
    W = nmf_model.fit_transform(A)
    H = nmf_model.components_

    return nmf_model, W, H


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


def generate_topics_from_NMF(H_matrix, index_to_word, top_n_words=15, print_out=False):
    """
    Create DataFrame where each row represents a "topic" from NMF
    Number of words for topic given by top top_n_words

    :param H_matrix: (numpy.ndarray) components matrix from NMF
    :param index_to_word: (dict) with key (int) index, value (str) word
    :param top_n_words: (int) number of words to show for given topic, def = 15
    :param print_out: (boolean) print while building list, def = False
    :return: pd.DataFrame of top_n_words terms for each topic
    """
    topic_list = []
    for topic_idx, topic in enumerate(H_matrix):
        top_n = [index_to_word[i] for i in topic.argsort()[-top_n_words:]][::-1]
        topic_list.append([topic_idx, top_n])
        if print_out:
            print(f"Topic {topic_idx}:\n{top_n}\n")
    return pd.DataFrame(topic_list, columns=["Topic", "Terms"])


def plot_top_words_with_weights_nmf(model_component, index_to_word, top_words=10):
    """
    Plot a horizontal bar chart
    Cretes one plot per topic, where each bar represents a top_n term in the topic
    The length of the bar is determined by its weight

    Input:
        :param model_component: (numpy.ndarray) components matrix from NMF
        :param index_to_word: (dict) with key (int) index, value (str) word
        :param top_n_words: (int) number of words to show for given topic, def = 15

    Return:
        NA, plot directly inline
    """
    for topic_idx, topic in enumerate(model_component):
        topic_copy = topic.copy()
        sorted_words = topic.argsort()[-top_words:][::-1]
        topic_copy.sort()
        topic_weights = topic_copy[-top_words:][::-1]
        topic_data = [[index_to_word[sorted_words[i]], weight]
                      for i, weight in enumerate(topic_weights)]
        topic_data = pd.DataFrame(topic_data, columns=["Term", "Weight"])

        plt.title(f'Top terms in NMF Topic {topic_idx}')
        sns.barplot(y="Term", x="Weight", data=topic_data, orient='h')
        plt.savefig(f"graphs/nmf_topic_{topic_idx}.jpeg", bbox_inches='tight')
        plt.show()
        plt.close()
