import re
import string
import spacy
from spacy.lang.en import stop_words as spacy_stopwords


nlp = spacy.load("en_core_web_lg",
                 disable=['parser', 'ner'])
STOPWORDS = spacy_stopwords.STOP_WORDS
PUNCTUATION = string.punctuation


def clean(text):
    """
    Cleans text string
        1. remove non alphanumeric characters
        2. replace numeric characters with word 'number'
        3. lower case
        3. stripws

    :param text: (str) input string
    :return: (str) clean string
    """
    # remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # replace numbers with the word 'number'
    text = re.sub(r"\d+", "number", text)
    # lower case
    text = text.lower()
    return text.strip()


def lemmatize(doc):
    """
    Lemmatize a spcy doc
    :param doc: spacy.tokens.doc.Doc
    :return: list of token lemma
    """
    lemma_list = [str(tok.lemma_) for tok in doc
                  if tok.is_alpha and len(tok) > 1 and tok.text not in STOPWORDS]
    return lemma_list


def tokenize(text):
    """
    tokenize and lemmatize text
    :param text: (str) input text
    :return: list (str), lemmatized tokens
    """
    tokenized_text = nlp(text)
    return lemmatize(tokenized_text)
