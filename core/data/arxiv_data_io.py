"""
Functions for creating arxiv dataset
"""
import json

CATEGORY_DICT = {
    "cs.AI": "Artificial Intelligence",
    "cs.CY": "Computers and Society",
    "cs.DM": "Discrete Mathematics",
    "q-bio.TO": "Tissues and Organs",
    "econ.TH": "Theoretical Economics",
    "eess.AS": "Audio and Speech Processing",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.PE ": "Populations and Evolution"
}


def create_set_for_category_dict(input_path, category_dict,
                                 single_cat_only=False,
                                 single_spec_cat=True):
    """
    Function to create subset of arxiv data with categories in category_dict

    If single_cat_only is True, only include articles with single cat.

    Otherwise (if single_cat_only is False),
        but singe_spec_cat is True, allow articles with multiple cats,
        only if it doesn't belong to two cats in category_dict

    If both are false allow articles with multiple cats, including
        multiple cats in category_dict

    :param input_path: path to full arxiv data (as string)
    :param category_dict: dict where key (str) is categories to include
    :param single_cat_only: if True only articles with single cat
    :param single_spec_cat: if True only articles with one of specified categories
    :return: list of dict where each dict represents an article
    """
    data_list = []
    cats = set(category_dict.keys())
    with open(input_path, 'r') as f:
        for line in f:
            article = json.loads(line)
            article_cats = set(article["categories"].split(" "))
            if single_cat_only and len(article_cats) > 1:
                continue
            if single_spec_cat and len(cats.intersection(article_cats)) > 1:
                continue
            for article_cat in article_cats:
                if article_cat in cats:
                    data_list.append(article)

    return data_list
