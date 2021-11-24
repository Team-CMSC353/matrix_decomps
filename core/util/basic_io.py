import json


def read_json_to_dict(file_path):
    """"
    read json file to dict obj
    input:
        (str) file_path: path to file
    output:
        (dict) data read from file
    """
    with open(file_path) as f:
        data_dict = json.load(f)
    return data_dict


def write_dict_to_json(output_file_path, data):
    """
    write dictionary to file
    input:
        (str) output_file_path: path to output file
        (dict) data: data to write
    """
    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile)
