# Applications of Matrix Factorizations for Unstructured Text

### Project Overview
The analysis of unstructured text documents often produces high-dimensional, complex datasets. Matrix factorizations provide simple techniques for performing dimensionality reduction and extracting meaningful features from the original text. We explore two methods for matrix factorization, Principal Components Analysis (PCA) and Non-negative Matrix Factorization (NMF), and apply them to a corpus of arXiv abstracts.
### Set up
_Currently using python 3.9.7_
1. Create virtual environment
`python3 -m venv env`
2. Activate `source env/bin/activate`
3. Install from dependencies `pip install -r requirements.txt`
4. (Optional) update dependencies `pip freeze > requirements.txt`

### Data
The data is obtained from the [arXiv dataset](https://www.kaggle.com/Cornell-University/arxiv/notebooks)

Our project specific data files can be downloaded #TODO-provide a link to a public folder

### Running the project
We provide jupyter notebooks view our analysis.
The notebooks can be found [here](https://github.com/Team-CMSC353/matrix_decomps/tree/main/notebooks).

To run the notebooks interactively, clone the repository and set up the virtual environment.
You can then start the jupyter notebook server using the following command `env/bin/jupyter-notebook`.

**Below is a brief summary of the notebooks**:

- **EDA - Full aXiv Data**: Basic summary of the 1.9M arXiv abstracts
- **EDA - Subset Category arXiv Data**: Summary of 12K arXiv abstracts used in analysis


