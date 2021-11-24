"""
Read arxiv data into dictionary
Key: (str) category
Value: (list of dict) list of article json

I don't recommend re-running this.
Instead use full_data_by_category.json.bz2:
https://drive.google.com/drive/folders/1mXg_ie1h0dwM1epOJyP779kbPi7JphtQ?usp=sharing
"""
import os
from core.util.basic_io import *

# download full dataset:
# https://drive.google.com/file/d/1yx3rnXbkhML-apiKLHbbPRCYpoEFnyoF/view?usp=sharing
# unzip into core/resources folder

file_name = "arxiv-metadata-oai-snapshot.json"
full_path = os.path.join("core", "resources", file_name)

data_by_category = {}
count = 1
with open(full_path, 'r') as f:
    for line in f:
        article = json.loads(line)
        categories_list = article["categories"].split(" ")
        for cat in categories_list:
            article_list = data_by_category.get(cat, [])
            article_list.append(article)
            data_by_category[cat] = article_list
        print("Processed: ", count)
        count += 1

output_file_name = "full_data_by_category.json"
output_full_path = os.path.join("core", "resources", output_file_name)
write_dict_to_json(output_full_path, data_by_category)
