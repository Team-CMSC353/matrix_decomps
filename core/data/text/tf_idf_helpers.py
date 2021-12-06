"""
Functions for training and applying TF-IDF vectorizer
"""
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# See documentation for scipy CSR sparse matrices
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix

def dummy_tokenizer(doc):
    """
    Dummy tokenizer to input into the TF-IDF Vectorizer function
    because we've already tokenized the data
    """
    return doc


def fit_tfidf(input_tokens,
            dummy=dummy_tokenizer):
    """
    Function to fit a tfid vectorizer object to an iterable of tokens

    :param input_tokens: iterable of lists of tokens, where each list corresponds
                            to a document
    :return tfidf: TfidVectorizer object that has been fit on input_tokens
    :return index_to_word: dict mapping index numbers to tokens -- contains every
                            unique token in the training corpora
    """
    tfidf = TfidfVectorizer(analyzer='word',
                            tokenizer=dummy,
                            preprocessor=dummy,
                            token_pattern=None)

    tfidf.fit(input_tokens)

    index_to_word = {index:word for word,index in tfidf.vocabulary_.items()}

    return tfidf, index_to_word


def transform_tfidf(input_df,
                   tfidf_obj,
                   token_col_name='tokens',
                   doc_id_col_name='id'):
    """
    Function to generate a tf_idf matrix for a corpora based on a tfidf
    Vectorizer object that has already been fit 

    :param input_df: pandas DataFrame containing corpora
    :param token_col_name: str name of the col containing tokens in input_df
    :param doc_id_col_name: str name of the col containing doc id in input_df
    :param tfidf_obj: TfidfVectorizer object that has already been
                        fit on some corpora
    :return tfidf_matrix: Compressed Sparse Row (CSR) tfidf matrix
    :return index_to_doc_id: dict mapping tfidf row indices to doc_ids
                                from input_df
    """
    index_to_doc_id = {index:doc_id for
                       index,doc_id in input_df[doc_id_col_name].items()}

    tfidf_matrix = tfidf_obj.transform(input_df[token_col_name])

    return tfidf_matrix, index_to_doc_id
