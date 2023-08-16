import argparse
import datetime
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
from results_analysis import run_full_analysis


CURR_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
METHODS_DIRECTORY = "methods/implemented_methods"
RESULTS_PATH = "results"
LOG_PATH = os.path.join(RESULTS_PATH, "logs", CURR_DATETIME)
LOG_METHOD_W_DATASET_PATH = os.path.join(RESULTS_PATH, "methods")


def main():
    config = get_config()

    if config.list_methods:
        experiments = scan_for_detection_methods()
        print_w_sep_line("Locally available methods:\n")
        for exp in experiments: print(exp.__name__)
        print_w_sep_line("Finish")
        exit(0)
    
    if config.name is not None:
        global LOG_PATH
        LOG_PATH = os.path.join(RESULTS_PATH, "logs", config.name)
    
    print_w_sep_line(f'Loading datasets {config.dataset_filepath}...')
    dataset_dict = load_multiple_from_file(filepaths=config.dataset_filepath, processors=config.dataset_processor, 
                                                          text_field=config.text_field, label_field=config.label_field, 
                                                          human_label=config.human_label, other=config.dataset_other)

    print_w_sep_line("Running benchmark...")
    benchmark_results = run_benchmark(dataset_dict, config)

    print_w_sep_line("Saving experiment results\n")
    log_whole_experiment(config, benchmark_results)    
    save_method_dataset_combination_results(config, benchmark_results)
    
    print_w_sep_line("Running analysis:\n")
    run_full_analysis(benchmark_results, LOG_PATH, config.interactive)
    
    print_w_sep_line("Finish")
                   
def run_benchmark(dataset_dict, config):
    
    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)
    print_w_sep_line(f"Using cache dir {config.cache_dir}")
    
    available_experiments = scan_for_detection_methods()    
    outputs = dict()
    
    for dataset_name, data in dataset_dict.items():
        print_w_sep_line(f"Running experiments on {dataset_name} dataset:\n")    
        outputs[dataset_name] = []
        for method in config.methods:
            try: 
                results = run_experiment(data, config, method, available_experiments)
            except Exception:
                print(f"Experiment {method} failed due to below reasons. Skipping and continuing with the next experiment.")
                print(traceback.format_exc())
                continue
            
            outputs[dataset_name].append(results)
    
    return outputs

def run_experiment(data, config, method, available_experiments):
    
    available_exp_names = list(map(lambda x: x.__name__, available_experiments))
    if method in available_exp_names:
        return available_experiments[available_exp_names.index(method)](data=data, config=config).run()
    
    try: # Check if method is model name from HuggingFace Hub for sequence classification
        return SupervisedExperiment(data, method, method, config).run()
    except:
        pass
    
    raise ValueError(f"Unknown method: {method}")
    

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
    main()
