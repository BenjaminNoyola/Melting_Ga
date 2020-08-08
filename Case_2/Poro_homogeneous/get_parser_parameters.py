import json
import argparse
import pprint

def setup_parser(arguments, title):

    parser = argparse.ArgumentParser(description=title,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, val in arguments.items():
        parser.add_argument('-%s' % key,
                            type=eval(val["type"]),
                            help=val["help"],
                            default=val["default"])
    return parser

def read_params(parser):
    parameters = vars(parser.parse_args())
    return parameters

def get_parameters(title=None):

    with open("Parameters.json") as data_file:
        data = json.load(data_file)
    parser = setup_parser(data, title)
    parameters = read_params(parser)
    return parameters

# if __name__ == "__main__":
#     params = get_parameters()
#     print(params["lattice_y"])
#     pprint.pprint(params)

