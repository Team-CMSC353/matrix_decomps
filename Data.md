# Instructions for recreating the data set
Most of the project notebooks require the following serialized data file `tokenized_arxiv_subset_15540.pkl`. This file contains the cleaned, tokenized version of our data set and can be downloaded #TODO.

Alternatively, the file can be recreated by running the following scripts (located in [`scripts`](https://github.com/Team-CMSC353/matrix_decomps/tree/main/scripts) folder)


## Process arXiv data, subset by category

- ### create_arxiv_data_set_for_analysis.py
    input: arxiv-metadata-oai-snapshot.json (from kaggle)
    
    output: arxiv_subset_15540.json

## Clean, tokenize, compute tf_idf

- ### clean_arxiv_data_set.py
   input: arxiv_subset_15540.json 

   output: tokenized_arxiv_subset_15540.pkl
