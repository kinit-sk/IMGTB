import argparse
import datetime
import os
import sys
import json
from pathlib import Path
from inspect import getmembers, getmodule, isclass
import importlib.util
import traceback

import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer, filter_test_data
from methods.abstract_methods.experiment import Experiment
from results_analysis import run_full_analysis

CURR_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
METHODS_DIRECTORY = "methods/implemented_methods"
RESULTS_PATH = "results"
LOG_PATH = os.path.join(RESULTS_PATH, "logs", CURR_DATETIME)
LOG_METHOD_W_DATASET_PATH = os.path.join(RESULTS_PATH, "methods")
SEP_LINE = "-----------------------------------------------------"


def scan_for_detection_methods():
    exp_class_list = []

    for file in os.scandir(METHODS_DIRECTORY):
        if not file.is_file() or not file.path.endswith(".py"):
            continue
        
        # Import the file to make its members accessible.
        spec = importlib.util.spec_from_file_location(file.name, file.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[file.name] = module
        spec.loader.exec_module(module)
        
        # Find all subclasses of Experiment.
        for _, obj in getmembers(module):
            if isclass(obj) and issubclass(obj, Experiment) and getmodule(obj) is module:
                exp_class_list.append(obj)

    return exp_class_list

def log_whole_experiment(args, outputs):
    """Log all experiment data as a whole by current time"""
        
    print(f"Logging all experiment data to absolute path: {os.path.abspath(LOG_PATH)}")
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    # Write args to file.
    with open(os.path.join(LOG_PATH, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4)
        
    # Save outputs.
    with open(os.path.join(LOG_PATH, f"benchmark_results.json"), "w") as file:
        json.dump(outputs, file)

def save_method_dataset_combination_results(args, outputs):
    """Log results for all method and dataset combinations separately"""

    for dataset_name, results_list in outputs.items():
        for method_data in results_list:
            method_name = method_data["name"]
            SAVE_PATH =  os.path.join(LOG_METHOD_W_DATASET_PATH, method_name, dataset_name)
            print(f"Saving results from {method_name}" 
                   f" on {dataset_name} dataset to absolute path: {os.path.abspath(SAVE_PATH)}")
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            data = {"args": args.__dict__, "data": method_data}
            # Save args and outputs for given method and dataset.
            with open(os.path.join(SAVE_PATH, f"{CURR_DATETIME}_experiment_results.json"), "w") as file:
                json.dump(data, file)

def print_w_sep_line(text: str) -> None:
    width = os.get_terminal_size().columns 
    print('-' * width)
    print(text)
               
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--dataset_filepath', nargs='+', type=str, default=["datasets/TruthfulQA_LMMs.csv"], 
                        help="List of datasets (filepaths)."
                             "(For datasets split across multiple files, specify only the directory where the files are all located.)")
    parser.add_argument('--dataset_processor', nargs='+', type=str, default=["default"], 
                        help="List of custom dataset processing functions corresponding to the dataset filepaths positionally.")
    parser.add_argument('--text_field', type=str, default="text", 
                        help="Name of the dataset column containing text.")
    parser.add_argument('--label_field', type=str, default="label", 
                        help="Name of the dataset column containing labels.")
    parser.add_argument('--human_label', type=str, default="0", 
                        help="String corresponding to the label marking human text.")
    parser.add_argument('--detectLLM', type=str, default="ChatGPT", 
                        help="For supported datasets, datasets with texts from multiple LLMs, "
                             "you can pass to the processor function against which LLM you want to run the benchmark.")

    # Use dataset_other to pass arbitrary text information from CLI to chosen dataset processor
    parser.add_argument('--dataset_other', nargs="+", type=str, 
                        help="Use to pass arbitrary text information to chosen dataset processor.")
    
    # List the methods you want to run
    # (methods are named after names of their respective classes in the methods/implemented_methods directory)
    parser.add_argument('--methods', nargs='+', type=str, default=["all"], help="List the names of methods you want to run.")
    parser.add_argument('--list_methods', action="store_true", help="List names of all available methods.")
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
    parser.add_argument('--DEVICE', type=str, default="cuda", help="Define a device to run the computations on (e.g. cuda, cpu...).")

    # params for DetectGPT
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

    # params for GPTZero
    parser.add_argument('--gptzero_key', type=str, default="")
    
    # results analysis params
    parser.add_argument('--list_analysis_methods', action='store_true')
    parser.add_argument('--analysis_methods', type=str, nargs='+', default=["all"])

    args = parser.parse_args()

    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')
    
    print_w_sep_line(f'Loading datasets {args.dataset_filepaths}...')
    dataset_dict = dataset_loader.load_multiple_from_file(filepaths=args.dataset_filepath, processors=args.dataset_processor, 
                                                          text_field=args.text_field, label_field=args.label_field, 
                                                          human_label=args.human_label, other=args.dataset_other)
    # data = filter_test_data(data, max_length=25)

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print_w_sep_line(f"Using cache dir {cache_dir}")

    # Get generative model
    print_w_sep_line(f"Loading BASE model {args.base_model_name}\n")
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.base_model_name, cache_dir)
    load_base_model(base_model, DEVICE)
    
    # Load experiments and evaluate them
    experiments = scan_for_detection_methods()
    if args.list_methods:
        print_w_sep_line("Available methods:\n")
        for exp in experiments: print(exp.__name__)
        print_w_sep_line("Finish")
        exit(0)
    
    outputs = dict()
    for dataset_name, data in dataset_dict.items():
        print_w_sep_line(f"Running experiments on {dataset_name} dataset:\n")    
        filtered = filter(lambda exp: args.methods[0] == "all" or exp.__name__ in args.methods, experiments)
        outputs[dataset_name] = []
        for experiment in filtered:
            try: 
                results = experiment(data=data, 
                                        model=base_model, 
                                        tokenizer=base_tokenizer, 
                                        DEVICE=DEVICE, 
                                        detectLLM=args.detectLLM, 
                                        batch_size=batch_size,
                                        cache_dir=cache_dir,
                                        args=args,
                                        gptzero_key=args.gptzero_key,
                                        clf_algo_for_threshold=args.clf_algo_for_threshold
                                        ).run()
            except Exception:
                print(f"Experiment {experiment.__name__} failed due to below reasons. Skipping and continuing with the next experiment.")
                print(traceback.format_exc())
                continue
            
            outputs[dataset_name].append(results)

    print_w_sep_line("Saving experiment results\n")
    log_whole_experiment(args, outputs)    
    save_method_dataset_combination_results(args, outputs)
    
    print_w_sep_line("Running analysis:\n")
    run_full_analysis(outputs, save_path=LOG_PATH)
    
    print_w_sep_line("Finish")
    