import os
import gensim.corpora as corpora
from core.data.arxiv_data_io import *
from core.data.text.tf_idf_helpers import *
from gensim.models.coherencemodel import CoherenceModel


class TrainingData:
    """
    Wrapper for training data and related matrices
    """

    def __init__(self, path_to_pkl_data):
        self.data_df = pd.read_pickle(path_to_pkl_data)

        self.train_df, _ = sample_arxiv_data_by_category(self.data_df)

        self.tokens = self.train_df['tokens']
        tfidf_obj, self.index_to_word = fit_tfidf(self.train_df['tokens'])
        self.tfidf_train_matrix, self.index_to_doc = transform_tfidf(self.train_df, tfidf_obj=tfidf_obj)

        # for coherence
        self.input_data = self.train_df['tokens'].tolist()
        self.id2word = corpora.Dictionary(self.input_data)
        self.corpus = [self.id2word.doc2bow(text) for text in self.input_data]

    def compute_coherence(self, topic_list):
        """
        Return coherence for topic model trained on self.train_df
        :param topic_list: list of list of terms
        :return: (float) coherence
        """

        cm = CoherenceModel(topics=topic_list, texts=self.input_data, corpus=self.corpus,
                            dictionary=self.id2word, coherence='c_v')
        return cm.get_coherence()
