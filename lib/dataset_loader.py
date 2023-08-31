import random
import datasets
import tqdm
import pandas as pd
import re
from typing import List, Dict, Union, Tuple
from pathlib import Path
from itertools import zip_longest
from sklearn.model_selection import train_test_split

""" 
    This module provides functionality for loading datasets
    
    It adheres, at least, to the following interface:
    
    load_multiple_from_file()
        - Returns a dictionary mapping dataset names to their processed data
    read_multiple_to_pandas()
        - Returns a dictionary mapping dataset names to their raw data (dictionary of one or more files and their data)
    read_dir_to_pandas()
        - Returns a dictionary mapping all files in the directory to their raw data
    read_file_to_pandas()
        - Returns a dictionary mapping the filename to its raw data
"""

SUPPORTED_FILETYPES = ["auto", "csv", "tsv", "xls", "xlsx", "json", "jsonl", "xml", "huggingfacehub"]


def load_multiple_from_file(datasets_params, is_interactive: bool):
    unified_data_dict = dict()
    dataset_dict = read_multiple_to_pandas(datasets_params, is_interactive)
    for dataset_data, dataset_params in zip(dataset_dict.items(), datasets_params):
        dataset_name, file_dict = dataset_data
        dataset_processor = globals().get(f'process_{dataset_params["processor"]}')
        if dataset_processor is not None:
            human_texts, machine_texts = dataset_processor(file_dict, 
                                                           dataset_params["text_field"], 
                                                           dataset_params["label_field"], 
                                                           dataset_params["human_label"], 
                                                           dataset_params["dataset_other"])
            unified_data_dict[dataset_name] = _data_to_unified(human_texts, machine_texts)
        else:
            raise ValueError(f'Unknown dataset processor: {processor}')
    
    return unified_data_dict

def read_multiple_to_pandas(datasets_params, is_interactive: bool) -> Dict[str, pd.DataFrame]:
    datasets = dict()
    for dataset_params in datasets_params:
        filepath, filetype = dataset_params["filepath"], dataset_params["filetype"]
        read_dataset_to_pandas = read_dir_to_pandas if Path(filepath).is_dir() else read_file_to_pandas
        while (df_dict := read_dataset_to_pandas(filepath, filetype, is_interactive)) is None:
            if is_interactive:
                filetype = input(f'Unknown dataset file format for: {filepath}. Please, input the correct file format manually.\n'
                                f'Options: {", ".join(SUPPORTED_FILETYPES)}\n')
                continue
            raise ValueError(f'Unknown dataset file format for: {filepath}')
        datasets[Path(filepath).stem] = df_dict
    
    return datasets

def read_dir_to_pandas(filepath: str, filetype: str, is_interactive: bool):
    data = dict()
    pathlist = Path(filepath).iterdir()
    for path in pathlist:
        while (df_dict := read_file_to_pandas(str(path), filetype)) is None:
            if is_interactive:
                filetype = input(f'Unknown dataset file format for: {filepath}. Please, input the correct file format manually.\n'
                                f'Options: {", ".join(SUPPORTED_FILETYPES)}\n')
                continue
            raise ValueError(f'Unknown dataset file format for: {filepath}')
        data.update(df_dict)
    
    return data
    

def read_file_to_pandas(filepath: str, filetype="auto", *args):
    data = dict()
    if filetype == "auto":
        filetype = filepath.rsplit(".", 1)[-1]
        
    match filetype:
        case "csv":
            data[Path(filepath).stem] = pd.read_csv(filepath)            
        case "tsv":
            data[Path(filepath).stem] = pd.read_csv(filepath, sep='\t')
        case "xls" | "xlsx":
            data[Path(filepath).stem] = pd.read_excel(filepath)
        case "json":
            data[Path(filepath).stem] = pd.read_json(filepath)
        case "jsonl":
            data[Path(filepath).stem] = pd.read_json(filepath, lines=True)
        case "xml":
            data[Path(filepath).stem] = pd.read_xml(filepath)
        case "huggingfacehub":
            data[Path(filepath).stem] = datasets.load_dataset(filepath).to_pandas()
        case _:
            return None
    
    return data 

#######################################################
#                     PROCESSORS                      #
#######################################################

def process_default(data: Dict[str, pd.DataFrame], text_field, label_field, human_label, *args) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    if text_field not in data.columns or label_field not in data.columns:
        raise ValueError("Could not parse dataset. Please, correctly specify dataset specifications or define your own custom function for parsing.")
    human_texts = data[text_field].where(data[label_field].astype("string") == human_label).dropna().reset_index(drop=True)
    machine_texts = data[text_field].where(data[label_field].astype("string") != human_label).dropna().reset_index(drop=True)
    return human_texts, machine_texts

def process_test_small_dir(data: Dict[str, pd.DataFrame], *args):
    human = data["human"]["human"]
    machine = data["machine"]["machine"]
    return human, machine
    

def process_TruthfulQA(data: Dict[str, pd.DataFrame], text_field, label_field, human_label, other) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = other
    a_human = data['Best Answer'].fillna("").where(1 < data['Best Answer'].str.split().apply(len)).astype(str)
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(1 < data[f'{detectLLM}_answer'].str.split().apply(len)).astype(str)
    return a_human, a_chat


def process_SQuAD1(data: Dict[str, pd.DataFrame], text_field, label_field, human_label, other) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = other
    a_human = [eval(ans)['text'][0] for ans in data['answers'].tolist() if len(eval(ans)['text'][0].split()) > 1]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(data[f'{detectLLM}_answer'].str.split().apply(len) > 1)
    return pd.Series(a_human), a_chat


def process_NarrativeQA(data: Dict[str, pd.DataFrame], text_field, label_field, human_label, other) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = other
    a_human = [ans.split(";")[0] for ans in data['answers'].tolist() if 1 < len(ans.split(";")[0].split()) < 150]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where((1 < data[f'{detectLLM}_answer'].str.split().apply(len)) & (data[f'{detectLLM}_answer'].str.split().apply(len) < 150))
    return pd.Series(a_human).astype(str), pd.Series(a_chat).astype(str)

def process_test_small(data: Dict[str, pd.DataFrame], *args) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    human_texts = data[data.label == 0]["text"]
    machine_texts = data[data.label == 1]["text"]
    return human_texts, machine_texts                       

###########################################################
#                         UTILS                           #
###########################################################

def _data_to_unified(human_texts: pd.Series, machine_texts: pd.Series) \
                -> Dict[str, Dict[str, Union[List[str], List[int]]]]:

    texts = pd.concat([human_texts, machine_texts], axis=0, names=["text"]).reset_index(drop=True)
    labels = pd.Series([0]*len(human_texts) + [1]*len(machine_texts))
    data = pd.DataFrame(data={"text": texts, "label": labels}).reset_index(drop=True)

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True, stratify=data["label"])

    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}


def _process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def _process_text_truthfulqa_adv(text):

    if "I am sorry" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    if "as an AI language model" in text or "As an AI language model" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    return text