import argparse
import datetime
from itertools import zip_longest
import os
import sys
import json
from pathlib import Path
from inspect import getmembers, getmodule, isclass
import importlib.util
import traceback

from lib.dataset_loader import load_multiple_from_file
from lib.config import get_config
from methods.abstract_methods.experiment import Experiment
from methods.abstract_methods.supervised_experiment import SupervisedExperiment
from results_analysis import run_full_analysis, list_available_analysis_methods


CURR_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
METHODS_DIRECTORY = "methods/implemented_methods"
RESULTS_PATH = "results"
LOG_PATH = os.path.join(RESULTS_PATH, "logs", CURR_DATETIME)
LOG_METHOD_W_DATASET_PATH = os.path.join(RESULTS_PATH, "methods")
DATASETS_PATH = "datasets"


def main():
    config = get_config()

    if config["global"]["list_methods"]:
        experiments = scan_for_detection_methods()
        print_w_sep_line("Locally available methods:\n", config["global"]["interactive"])
        for exp in experiments: print(exp.__name__)
        print_w_sep_line("Finish", config["global"]["interactive"])
        exit(0)
    
    if config["global"]["list_datasets"]:
        print_w_sep_line("Available datasets:\n", config["global"]["interactive"])
        pathlist = Path(DATASETS_PATH).iterdir()
        for path in pathlist:
            print(path.stem)
        print_w_sep_line("Finish", config["global"]["interactive"])
        exit(0)

    if config["global"]["list_analysis_methods"]:
        print_w_sep_line("Available analysis methods:\n", config["global"]["interactive"])
        list_available_analysis_methods()
        print_w_sep_line("Finish", config["global"]["interactive"])
        exit(0)

    if config["global"]["name"] is not None:
        global LOG_PATH
        LOG_PATH = os.path.join(RESULTS_PATH, "logs", config["global"]["name"])
    
    print_w_sep_line(f"Loading datasets {[list(dataset.values())[0] for dataset in config['data']['list']]}...", config["global"]["interactive"])
    dataset_dict = load_multiple_from_file(datasets_params=config["data"]["list"], is_interactive=config["global"]["interactive"])

    print_w_sep_line("Running benchmark...", config["global"]["interactive"])
    benchmark_results = run_benchmark(dataset_dict, config)

    print_w_sep_line("Saving experiment results\n", config["global"]["interactive"])
    log_whole_experiment(config, benchmark_results)    
    save_method_dataset_combination_results(config["methods"]["list"], benchmark_results)
    
    print_w_sep_line("Running analysis:\n", config["global"]["interactive"])
    run_full_analysis(benchmark_results, config["analysis"], LOG_PATH, config["global"]["interactive"])
    
    print_w_sep_line("Finish", config["global"]["interactive"])
                   
def run_benchmark(dataset_dict, config):
    
    available_experiments = scan_for_detection_methods()    
    outputs = dict()
    
    for dataset_name, data in dataset_dict.items():
        print_w_sep_line(f"Running experiments on {dataset_name} dataset:\n", config["global"]["interactive"])    
        outputs[dataset_name] = []
        
        if config["methods"]["list"][0]["name"] == "all":
            outputs[dataset_name] = run_all_available(data, config["methods"]["list"][0], available_experiments)
            continue
            
        for method_config in config["methods"]["list"]:
            try:
                results = run_experiment(data, method_config, method_config["name"], available_experiments)
            except Exception:
                print(f"Experiment {method_config['name']} failed due to below reasons. Skipping and continuing with the next experiment.")
                print(traceback.format_exc())
                continue
            
            outputs[dataset_name].append(results)
    
    return outputs

def run_all_available(data, method_config, available_experiments):
    results = []
    for experiment in available_experiments:
        try:
            results.append(experiment(data=data, config=method_config).run())
        except Exception:
            print(f"Experiment {method_config['name']} failed due to below reasons. Skipping and continuing with the next experiment.")
            print(traceback.format_exc())
            continue
    return results

def run_experiment(data, method_config, method_name, available_experiments):
    
    available_exp_names = list(map(lambda x: x.__name__, available_experiments))
    if method_name in available_exp_names:
        return available_experiments[available_exp_names.index(method_name)](data=data, config=method_config).run()
    
    try: # Check if method is model name from HuggingFace Hub for sequence classification
        return SupervisedExperiment(data, method_name, method_name, method_config).run()
    except:
        print(traceback.format_exc())
        pass
    
    raise ValueError(f"Unknown method: {method_name}")
    

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


def log_whole_experiment(config, outputs):
    """Log all experiment data as a whole by current time"""
        
    print(f"Logging all experiment data to absolute path: {os.path.abspath(LOG_PATH)}")
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    # Write args to file.
    with open(os.path.join(LOG_PATH, "config.json"), "w") as file:
        json.dump(config, file, indent=4)
        
    # Save outputs.
    with open(os.path.join(LOG_PATH, f"benchmark_results.json"), "w") as file:
        json.dump(outputs, file)


def save_method_dataset_combination_results(methods_config, outputs):
    """Log results for all method and dataset combinations separately"""
    is_all = False
    if methods_config[0]["name"] == "all":
        is_all = True
    
    for dataset_name, results_list in outputs.items():
        for method_results, method_config in zip_longest(results_list, methods_config):
            if method_results is None:
                continue
            method_name = method_results["name"].replace("/", "-") # slash would create nested directory
            if is_all:
                method_config = methods_config[0]
            SAVE_PATH =  os.path.join(LOG_METHOD_W_DATASET_PATH, method_name, dataset_name)
            print(f"Saving results from {method_name}" 
                   f" on {dataset_name} dataset to absolute path: {os.path.abspath(SAVE_PATH)}")
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            data = {"config": method_config, "data": method_results}
            # Save args and outputs for given method and dataset.
            with open(os.path.join(SAVE_PATH, f"{CURR_DATETIME}_experiment_results.json"), "w") as file:
                json.dump(data, file)


def print_w_sep_line(text: str, is_interactive=True) -> None:
    width = 80
    if is_interactive:
        width = os.get_terminal_size().columns 
    print('-' * width)
    print(text)
    

if __name__ == '__main__':
    main()
