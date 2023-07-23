import argparse
import datetime
import os
import sys
import json
from pathlib import Path
from inspect import getmembers, getmodule, isclass
import importlib.util
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer, filter_test_data
from methods.abstract_methods.experiment import Experiment

METHODS_DIRECTORY = "methods/implemented_methods"

def scan_for_detection_methods():
    obj_list = []

    for file in os.scandir(METHODS_DIRECTORY):
        if not file.is_file() or not file.path.endswith(".py"):
            continue
        spec = importlib.util.spec_from_file_location(file.name, file.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[file.name] = module
        spec.loader.exec_module(module)
        for _, obj in getmembers(module):
            if isclass(obj) and issubclass(obj, Experiment) and getmodule(obj) is module:
                obj_list.append(obj)
            
    return obj_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--dataset_filepaths', nargs='+', type=str, default=["datasets/TruthfulQA_LMMs.csv"])
    parser.add_argument('--dataset_filetypes', nargs='+', type=str, default=["auto"], choices=["auto", "csv", "xls", "xlsx", "json", "xml", "huggingface"])
    parser.add_argument('--dataset_processor', type=str, required=True)
    # Use dataset_other to pass arbitrary text information from CLI to chosen dataset processor
    parser.add_argument('--dataset_other', nargs="+", type=str)
    
    # List the methods you want to run
    # (methods are named after names of their respective classes in the methods/implemented_methods directory)
    parser.add_argument('--methods', nargs='+', type=str, default=["all"])
    parser.add_argument('--detectLLM', type=str, default="ChatGPT")
    
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

    args = parser.parse_args()

    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    print(f'Loading datasets {args.dataset_filepaths}...')
    data = dataset_loader.load_from_file(args.dataset_filepaths, args.dataset_filetypes, 
                                         args.dataset_processor, args.dataset_other)
    # data = filter_test_data(data, max_length=25)

    base_model_name = args.base_model_name.replace('/', '_')
    SAVE_PATH = f"results/{base_model_name}-{args.mask_filling_model_name}/{args.dataset_processor}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

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
    
    experiments = scan_for_detection_methods()
    filtered = filter(lambda exp: args.methods[0] == "all" or exp.__name__ in args.methods, experiments)
    outputs = list(map(lambda obj: obj(data=data, 
                                       model=base_model, 
                                       tokenizer=base_tokenizer, 
                                       DEVICE=DEVICE, 
                                       detectLLM=args.detectLLM, 
                                       batch_size=batch_size,
                                       cache_dir=cache_dir,
                                       args=args,
                                       gptzero_key=args.gptzero_key
                                       ).run(), filtered))

    # # run GPTZero: pleaze specify your gptzero_key in the args
    # outputs.append(run_gptzero_experiment(data, api_key=args.gptzero_key))

    # run DetectGPT
    #outputs.append(run_detectgpt_experiments(
    #    args, data, base_model, base_tokenizer))

    # save results
    import pickle as pkl
    with open(os.path.join(SAVE_PATH, f"benchmark_results.pkl"), "wb") as f:
        pkl.dump(outputs, f)

    with open("logs/performance.csv", "a") as wf:
        for row in outputs:
            wf.write(f"{args.dataset_processor},{args.detectLLM},{args.base_model_name},{row['name']},{json.dumps(row['general'])}\n")

    print("Finish")
    