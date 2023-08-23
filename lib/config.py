import argparse
from lib.dataset_loader import SUPPORTED_FILETYPES

"""
    This module provides functionality for aquiring input parameters (e.g. through command-line arguments, YAML config file, interactive questioning or other).

    It adheres to the following interface:
    
    get_config()
        - This function returns a dictionary structure (similar to the YAML format) of all input parameters
"""

DEFAULT_DATASET_FILEPATH = "datasets/test_small.csv"
DEFAULT_DATASET_FILETYPE = "auto"
DEFAULT_DATASET_PROCESSOR = "default"
DEFAULT_DATASET_TEXT_FIELD = "text"
DEFAULT_DATASET_LABEL_FIELD = "label"
DEFAULT_DATASET_HUMAN_LABEL = "0"
DEFAULT_DATASET_OTHER = None


def get_config():
    return _parse_cmd_args()


def _parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--interactive', action="store_true")
    parser.add_argument('--name', type=str, default=None,
                        help="Set a custom name for the results log save file.")
    parser.add_argument('--from_config', type=str, default=None,
                        help="Specify filepath to YAML config file from which to read all parameters instead of the command-line arguments")

    # Parameters for dataset loading/parsing
    parser.add_argument('--dataset', nargs='+', action=_DatasetAppendAction, type=str, default=[(DEFAULT_DATASET_FILEPATH,
                                                                                                 DEFAULT_DATASET_FILETYPE,
                                                                                                 DEFAULT_DATASET_PROCESSOR,
                                                                                                 DEFAULT_DATASET_TEXT_FIELD, 
                                                                                                 DEFAULT_DATASET_LABEL_FIELD, 
                                                                                                 DEFAULT_DATASET_HUMAN_LABEL, 
                                                                                                 DEFAULT_DATASET_OTHER)],
                        help="Define all dataset parameters.\n"
                             "Usage: --dataset FILEPATH FILETYPE PROCESSOR TEXT_FIELD LABEL_FIELD HUMAN_LABEL OTHER\n"
                             "Only required parameter is the dataset filepath, other parameters will be filled in with their default values."
                  s           )

    # List the methods you want to run
    # (methods are named after names of their respective classes in the methods/implemented_methods directory)
    parser.add_argument('--methods', nargs='+', type=str,
                        default=["all"], help="List the names of methods you want to run.")
    parser.add_argument('--list_methods', action="store_true",
                        help="List names of all available methods.")
    # Select an algorithm that will be used for threshold computation (for metric-based methods)
    # (You can define your own in methods/utils.py source file by creating a new item in the CLF_MODELS dictionary)
    parser.add_argument('--clf_algo_for_threshold', type=str, default="LogisticRegression",
                        choices=["LogisticRegression",
                                 "KNeighborsClassifier",
                                 "SVC",
                                 "DecisionTreeClassifier",
                                 "RandomForestClassifier",
                                 "MLPClassifier",
                                 "AdaBoostClassifier"],
                        help="Specify a classification algorithm to be used for threshold computation.")

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name',
                        type=str, default="t5-large")
    parser.add_argument('--cache_dir', type=str, default=".cache")
    parser.add_argument('--DEVICE', type=str, default="cuda",
                        help="Define a device to run the computations on (e.g. cuda, cpu...).")

    # Parameters for DetectGPT detection method
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbation_list', type=str, default="10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')

    # Parameters for GPTZero detection method
    parser.add_argument('--gptzero_key', type=str, default="")

    # Parameters for automatic benchmark results analysis
    parser.add_argument('--list_analysis_methods', action='store_true')
    parser.add_argument('--analysis_methods', type=str,
                        nargs='+', default=["all"])

    args = parser.parse_args()
    # Hotfix for argparse issue:
    # https://bugs.python.org/issue16399
    # removes default value, if different value/s has been specified by user
    if len(args.dataset) != 1:
        args.dataset = args.dataset[1:]

    return args


def _parse_dataset_args_by_length(values):
    if len(values) == 1:
        return (values[0], DEFAULT_DATASET_FILETYPE, DEFAULT_DATASET_PROCESSOR, DEFAULT_DATASET_TEXT_FIELD, DEFAULT_DATASET_LABEL_FIELD, DEFAULT_DATASET_HUMAN_LABEL, DEFAULT_DATASET_OTHER)
    if len(values) == 2:
        return (values[0], values[1], DEFAULT_DATASET_PROCESSOR, DEFAULT_DATASET_TEXT_FIELD, DEFAULT_DATASET_LABEL_FIELD, DEFAULT_DATASET_HUMAN_LABEL, DEFAULT_DATASET_OTHER)        
    if len(values) == 3:
        return (values[0], values[1], values[2], DEFAULT_DATASET_TEXT_FIELD, DEFAULT_DATASET_LABEL_FIELD, DEFAULT_DATASET_HUMAN_LABEL, DEFAULT_DATASET_OTHER)        
    if len(values) == 4:
        return (values[0], values[1], values[2], values[3], DEFAULT_DATASET_LABEL_FIELD, DEFAULT_DATASET_HUMAN_LABEL, DEFAULT_DATASET_OTHER)
    if len(values) == 5:
        return (values[0], values[1], values[2], values[3], values[4], DEFAULT_DATASET_HUMAN_LABEL, DEFAULT_DATASET_OTHER)        
    if len(values) == 6:
        return (values[0], values[1], values[2], values[3], values[4], values[5], DEFAULT_DATASET_OTHER) 
    if len(values) == 7:
        return (values[0], values[1], values[2], values[3], values[4], values[5], values[6])
    raise ValueError(
        f"--dataset option must have at most 3 arguments, provided: {values}")


class _DatasetAppendAction(argparse.Action):

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        if nargs == 0:
            raise ValueError('nargs for append actions must be != 0; if arg '
                             'strings are not supplying the value to append, '
                             'the append const action may be more appropriate')
        if const is not None and nargs != '?':
            raise ValueError('nargs must be %r to supply const' % '?')
        super(_DatasetAppendAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items.append(_parse_dataset_args_by_length(values))
        setattr(namespace, self.dest, items)


def _from_yaml_config(filepath: str):
    pass
