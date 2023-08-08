import argparse
import datetime
import os
import sys
import json
from pathlib import Path
from inspect import getmembers, getmodule, isclass
import importlib.util
import pickle as pkl

import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer, filter_test_data
from methods.abstract_methods.experiment import Experiment
from results_analysis import run_full_analysis

CURR_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
METHODS_DIRECTORY = "methods/implemented_methods"
RESULTS_PATH = "results"
LOG_PATH = os.path.join(RESULTS_PATH, "logs", CURR_DATETIME)
LOG_METHOD_W_DATASET_PATH = os.path.join(RESULTS_PATH, "methods")


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
        
    print(f"Saving results to absolute path: {os.path.abspath(LOG_PATH)}")
    print(LOG_PATH)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    # Write args to file.
    with open(os.path.join(LOG_PATH, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4)
        
    # Save outputs.
    with open(os.path.join(LOG_PATH, f"benchmark_results.pkl"), "wb") as file:
        pkl.dump(outputs, file)

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
            with open(os.path.join(SAVE_PATH, f"{CURR_DATETIME}_experiment_results.pkl"), "wb") as file:
                pkl.dump(outputs, file)
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--dataset_filepaths', nargs='+', type=str, default=["datasets/TruthfulQA_LMMs.csv"])
    parser.add_argument('--dataset_processors', nargs='+', type=str, default=["default"])
    parser.add_argument('--text_field', type=str, default="text")
    parser.add_argument('--label_field', type=str, default="label")
    parser.add_argument('--human_label', type=str, default="0")
    parser.add_argument('--detectLLM', type=str, default="ChatGPT")

    # Use dataset_other to pass arbitrary text information from CLI to chosen dataset processor
    parser.add_argument('--dataset_other', nargs="+", type=str)
    
    # List the methods you want to run
    # (methods are named after names of their respective classes in the methods/implemented_methods directory)
    parser.add_argument('--methods', nargs='+', type=str, default=["all"])
    parser.add_argument('--list_methods', action="store_true")
    # Select an algorithm that will be used for threshold computation (for metric-based methods)
    # (You can define your own in methods/utils.py source file by creating a new item in the CLF_MODELS dictionary)
    parser.add_argument('--clf_algo_for_threshold', type=str, default="LogisticRegression", choices=["LogisticRegression", 
                                                                                                     "KNeighborsClassifier", 
                                                                                                     "SVC", 
                                                                                                     "DecisionTreeClassifier", 
                                                                                                     "RandomForestClassifier", 
                                                                                                     "MLPClassifier", 
                                                                                                     "AdaBoostClassifier"])
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name',
                        type=str, default="t5-large")
    parser.add_argument('--cache_dir', type=str, default=".cache")
    parser.add_argument('--DEVICE', type=str, default="cuda")

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
    
    print(f'Loading datasets {args.dataset_filepaths}...')
    dataset_dict = dataset_loader.load_multiple_from_file(filepaths=args.dataset_filepaths, processors=args.dataset_processors, 
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
    print(f"Using cache dir {cache_dir}")

    # get generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.base_model_name, cache_dir)
    load_base_model(base_model, DEVICE)
    
    # Load experiments and evaluate them
    experiments = scan_for_detection_methods()
    if args.list_methods:
        print("\nAvailable methods:\n")
        for exp in experiments: print(exp.__name__)
        print("\nFinish")
        exit(0)
    
    outputs = dict()
    for dataset_name, data in dataset_dict.items():
        print(f"Running experiments on {dataset_name} dataset:")    
        filtered = filter(lambda exp: args.methods[0] == "all" or exp.__name__ in args.methods, experiments)
        outputs[dataset_name] = list(map(lambda obj: obj(data=data, 
                                            model=base_model, 
                                            tokenizer=base_tokenizer, 
                                            DEVICE=DEVICE, 
                                            detectLLM=args.detectLLM, 
                                            batch_size=batch_size,
                                            cache_dir=cache_dir,
                                            args=args,
                                            gptzero_key=args.gptzero_key,
                                            clf_algo_for_threshold=args.clf_algo_for_threshold
                                            ).run(), filtered))

    log_whole_experiment(args, outputs)    
    save_method_dataset_combination_results(args, outputs)
    
    print("\nRunning analysis:")
    run_full_analysis(outputs, save_path=LOG_PATH)
    
    print("Finish")
    