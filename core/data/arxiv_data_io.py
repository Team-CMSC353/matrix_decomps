"""
Functions for creating arxiv dataset
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ID_KEY = "id"
AUTHORS_KEY = "authors"
TITLE_KEY = "title"
CAT_KEY = "categories"
ABSTRACT_KEY = "abstract"
UPDATE_DT_KEY = 'update_date'

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


def create_arxiv_df(arxiv_dicts):
    """
    Convert list of json articles to pd DataFrame
    :param arxiv_dicts: list of dict where each dict is article
    :return: pd DataFrame
    """
    data_list = []
    for article in arxiv_dicts:
        id = article[ID_KEY]
        authors = article[AUTHORS_KEY]
        title = article[TITLE_KEY]
        categories = article[CAT_KEY]
        abstract = article[ABSTRACT_KEY]
        update_dt = article[UPDATE_DT_KEY]
        data_list.append([id, authors, title,
                          categories, abstract, update_dt])

    category_df = pd.DataFrame(data_list,
                               columns=['id', 'authors', 'title',
                                        'categories', 'abstract', 'update_dt'])
    return category_df


def sample_arxiv_data_by_category(arxiv_df):
    """
    Sample arxiv data, using 80% of data for training.
    Returns tuple of DataFrames

    :param arxiv_df: pandas DataFrame
    :return: Tuple of DataFrames representing train and test data
    """

    sample_train, sample_test = train_test_split(arxiv_df, train_size=0.8,
                                                random_state=3020211120,
                                                stratify=arxiv_df["categories"])

    sample_train['full_df_index'] = sample_train.index
    sample_test['full_df_index'] = sample_test.index

    sample_train.reset_index(inplace=True)
    sample_test.reset_index(inplace=True)

    return (sample_train, sample_test)


# from https://www.kaggle.com/artgor/arxiv-metadata-exploration
# https://arxiv.org/help/api/user-manual
category_map = {'astro-ph': 'Astrophysics',
                'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
                'astro-ph.EP': 'Earth and Planetary Astrophysics',
                'astro-ph.GA': 'Astrophysics of Galaxies',
                'astro-ph.HE': 'High Energy Astrophysical Phenomena',
                'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
                'astro-ph.SR': 'Solar and Stellar Astrophysics',
                'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
                'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
                'cond-mat.mtrl-sci': 'Materials Science',
                'cond-mat.other': 'Other Condensed Matter',
                'cond-mat.quant-gas': 'Quantum Gases',
                'cond-mat.soft': 'Soft Condensed Matter',
                'cond-mat.stat-mech': 'Statistical Mechanics',
                'cond-mat.str-el': 'Strongly Correlated Electrons',
                'cond-mat.supr-con': 'Superconductivity',
                'cs.AI': 'Artificial Intelligence',
                'cs.AR': 'Hardware Architecture',
                'cs.CC': 'Computational Complexity',
                'cs.CE': 'Computational Engineering, Finance, and Science',
                'cs.CG': 'Computational Geometry',
                'cs.CL': 'Computation and Language',
                'cs.CR': 'Cryptography and Security',
                'cs.CV': 'Computer Vision and Pattern Recognition',
                'cs.CY': 'Computers and Society',
                'cs.DB': 'Databases',
                'cs.DC': 'Distributed, Parallel, and Cluster Computing',
                'cs.DL': 'Digital Libraries',
                'cs.DM': 'Discrete Mathematics',
                'cs.DS': 'Data Structures and Algorithms',
                'cs.ET': 'Emerging Technologies',
                'cs.FL': 'Formal Languages and Automata Theory',
                'cs.GL': 'General Literature',
                'cs.GR': 'Graphics',
                'cs.GT': 'Computer Science and Game Theory',
                'cs.HC': 'Human-Computer Interaction',
                'cs.IR': 'Information Retrieval',
                'cs.IT': 'Information Theory',
                'cs.LG': 'Machine Learning',
                'cs.LO': 'Logic in Computer Science',
                'cs.MA': 'Multiagent Systems',
                'cs.MM': 'Multimedia',
                'cs.MS': 'Mathematical Software',
                'cs.NA': 'Numerical Analysis',
                'cs.NE': 'Neural and Evolutionary Computing',
                'cs.NI': 'Networking and Internet Architecture',
                'cs.OH': 'Other Computer Science',
                'cs.OS': 'Operating Systems',
                'cs.PF': 'Performance',
                'cs.PL': 'Programming Languages',
                'cs.RO': 'Robotics',
                'cs.SC': 'Symbolic Computation',
                'cs.SD': 'Sound',
                'cs.SE': 'Software Engineering',
                'cs.SI': 'Social and Information Networks',
                'cs.SY': 'Systems and Control',
                'econ.EM': 'Econometrics',
                'eess.AS': 'Audio and Speech Processing',
                'eess.IV': 'Image and Video Processing',
                'eess.SP': 'Signal Processing',
                'gr-qc': 'General Relativity and Quantum Cosmology',
                'hep-ex': 'High Energy Physics - Experiment',
                'hep-lat': 'High Energy Physics - Lattice',
                'hep-ph': 'High Energy Physics - Phenomenology',
                'hep-th': 'High Energy Physics - Theory',
                'math.AC': 'Commutative Algebra',
                'math.AG': 'Algebraic Geometry',
                'math.AP': 'Analysis of PDEs',
                'math.AT': 'Algebraic Topology',
                'math.CA': 'Classical Analysis and ODEs',
                'math.CO': 'Combinatorics',
                'math.CT': 'Category Theory',
                'math.CV': 'Complex Variables',
                'math.DG': 'Differential Geometry',
                'math.DS': 'Dynamical Systems',
                'math.FA': 'Functional Analysis',
                'math.GM': 'General Mathematics',
                'math.GN': 'General Topology',
                'math.GR': 'Group Theory',
                'math.GT': 'Geometric Topology',
                'math.HO': 'History and Overview',
                'math.IT': 'Information Theory',
                'math.KT': 'K-Theory and Homology',
                'math.LO': 'Logic',
                'math.MG': 'Metric Geometry',
                'math.MP': 'Mathematical Physics',
                'math.NA': 'Numerical Analysis',
                'math.NT': 'Number Theory',
                'math.OA': 'Operator Algebras',
                'math.OC': 'Optimization and Control',
                'math.PR': 'Probability',
                'math.QA': 'Quantum Algebra',
                'math.RA': 'Rings and Algebras',
                'math.RT': 'Representation Theory',
                'math.SG': 'Symplectic Geometry',
                'math.SP': 'Spectral Theory',
                'math.ST': 'Statistics Theory',
                'math-ph': 'Mathematical Physics',
                'nlin.AO': 'Adaptation and Self-Organizing Systems',
                'nlin.CD': 'Chaotic Dynamics',
                'nlin.CG': 'Cellular Automata and Lattice Gases',
                'nlin.PS': 'Pattern Formation and Solitons',
                'nlin.SI': 'Exactly Solvable and Integrable Systems',
                'nucl-ex': 'Nuclear Experiment',
                'nucl-th': 'Nuclear Theory',
                'physics.acc-ph': 'Accelerator Physics',
                'physics.ao-ph': 'Atmospheric and Oceanic Physics',
                'physics.app-ph': 'Applied Physics',
                'physics.atm-clus': 'Atomic and Molecular Clusters',
                'physics.atom-ph': 'Atomic Physics',
                'physics.bio-ph': 'Biological Physics',
                'physics.chem-ph': 'Chemical Physics',
                'physics.class-ph': 'Classical Physics',
                'physics.comp-ph': 'Computational Physics',
                'physics.data-an': 'Data Analysis, Statistics and Probability',
                'physics.ed-ph': 'Physics Education',
                'physics.flu-dyn': 'Fluid Dynamics',
                'physics.gen-ph': 'General Physics',
                'physics.geo-ph': 'Geophysics',
                'physics.hist-ph': 'History and Philosophy of Physics',
                'physics.ins-det': 'Instrumentation and Detectors',
                'physics.med-ph': 'Medical Physics',
                'physics.optics': 'Optics',
                'physics.plasm-ph': 'Plasma Physics',
                'physics.pop-ph': 'Popular Physics',
                'physics.soc-ph': 'Physics and Society',
                'physics.space-ph': 'Space Physics',
                'q-bio.BM': 'Biomolecules',
                'q-bio.CB': 'Cell Behavior',
                'q-bio.GN': 'Genomics',
                'q-bio.MN': 'Molecular Networks',
                'q-bio.NC': 'Neurons and Cognition',
                'q-bio.OT': 'Other Quantitative Biology',
                'q-bio.PE': 'Populations and Evolution',
                'q-bio.QM': 'Quantitative Methods',
                'q-bio.SC': 'Subcellular Processes',
                'q-bio.TO': 'Tissues and Organs',
                'q-fin.CP': 'Computational Finance',
                'q-fin.EC': 'Economics',
                'q-fin.GN': 'General Finance',
                'q-fin.MF': 'Mathematical Finance',
                'q-fin.PM': 'Portfolio Management',
                'q-fin.PR': 'Pricing of Securities',
                'q-fin.RM': 'Risk Management',
                'q-fin.ST': 'Statistical Finance',
                'q-fin.TR': 'Trading and Market Microstructure',
                'quant-ph': 'Quantum Physics',
                'stat.AP': 'Applications',
                'stat.CO': 'Computation',
                'stat.ME': 'Methodology',
                'stat.ML': 'Machine Learning',
                'stat.OT': 'Other Statistics',
                'stat.TH': 'Statistics Theory'}
