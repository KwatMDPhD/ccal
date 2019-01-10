from json import load

from pandas import DataFrame


def make_case_annotations(json_file_path):

    with open(json_file_path) as json_file:

        cases = load(json_file)

    for case in cases:

        case.pop("summary")

        for key in ("demographic", "project"):

            if key in case:

                for k, v in case.pop(key).items():

                    case[k] = v

    return DataFrame(cases).set_index("submitter_id")
