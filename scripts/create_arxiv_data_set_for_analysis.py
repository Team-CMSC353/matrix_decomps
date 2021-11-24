"""
Create subset of arxiv data
Categories defined in core.data.arxiv_data_io
"""
import os
from core.data.arxiv_data_io import create_set_for_category_dict, CATEGORY_DICT
from core.util.basic_io import write_dict_to_json


file_name = "arxiv-metadata-oai-snapshot.json"
full_path = os.path.join("core", "resources", file_name)
data = create_set_for_category_dict(full_path, CATEGORY_DICT,
                                    single_cat_only=True,
                                    single_spec_cat=True)

output_file_name = f"arxiv_subset_{round(len(data),-1)}.json"
output_full_path = os.path.join("scripts", "output", output_file_name)
write_dict_to_json(output_full_path, data)