from json import dump, load


def read_json(json_file_path):

    with open(json_file_path) as json_file:

        return load(json_file)


def write_json(json_dict, json_file_path, indent=2):

    with open(json_file_path, "w") as json_file:

        dump(json_dict, json_file, indent=indent)
