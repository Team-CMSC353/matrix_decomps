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

CATEGORY_DESCR = {
    "cs.AI": "Covers all areas of AI except Vision, Robotics, Machine Learning,"
             "Multiagent Systems, and Computation and Language (Natural Language Processing), "
             "which have separate subject areas. In particular, includes Expert Systems, "
             "Theorem Proving (although this may overlap with Logic in Computer Science), "
             "Knowledge Representation, Planning, and Uncertainty in AI.",

    "cs.CY": "Covers impact of computers on society, computer ethics, "
             "information technology and public policy, legal aspects of computing, computers and education.",

    "cs.DM": "Covers combinatorics, graph theory, applications of probability.",

    "q-bio.TO": "Blood flow in vessels, biomechanics of bones, "
                "electrical waves, endocrine system, tumor growth",

    "econ.TH": "Includes theoretical contributions to Contract Theory, Decision Theory, "
               "Game Theory, General Equilibrium, Growth, Learning and Evolution, "
               "Macroeconomics, Market and Mechanism Design, and Social Choice.",

    "eess.AS": "Theory and methods for processing signals representing audio, "
               "speech, and language, and their applications. "
               "This includes analysis, synthesis, enhancement, transformation, "
               "classification and interpretation of such signals as well as the design, "
               "development, and evaluation of associated signal processing systems. "
               "Machine learning and pattern analysis applied to any of the above areas is also welcome.",

    "q-bio.NC": "Synapse, cortex, neuronal dynamics, neural network, "
                "sensorimotor control, behavior, attention",

    "q-bio.PE ": "Population dynamics, spatio-temporal and epidemiological models, "
                 "dynamic speciation, co-evolution, biodiversity, foodwebs, aging; "
                 "molecular evolution and phylogeny; directed evolution; origin of life"
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
