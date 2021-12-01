"""
Functions for training and applying TF-IDF vectorizer
"""
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def dummy_tokenizer(doc):
    """
    Dummy tokenizer to input into the TF-IDF Vectorizer function
    because we've already tokenized the data
    """
    return doc


def fit_tfid(input_tokens,dummy=dummy_tokenizer):
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