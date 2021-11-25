import os
from core.util.basic_io import *
from core.data.arxiv_data_io import *
from core.data.text.cleaning import *

file_name = "arxiv_subset_15540.json"
full_path = os.path.join("core", "resources", file_name)
data = read_json_to_dict(full_path)

data_df = create_arxiv_df(data)
data_df['clean'] = data_df['abstract'].apply(clean)
data_df['tokens'] = data_df['clean'].apply(tokenize)

output_file_name = f"tokenized_arxiv_subset_{round(len(data_df),-1)}.pkl"
output_full_path = os.path.join("scripts", "output", output_file_name)
data_df.to_pickle(output_full_path)
