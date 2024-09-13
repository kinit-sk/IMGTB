import random
import datasets
import tqdm
import pandas as pd
import re
from typing import List, Dict, Union, Tuple
from pathlib import Path
from itertools import zip_longest
from sklearn.model_selection import train_test_split
from math import floor

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
DEFAULT_TEXT_FIELD_NAME = "text"
DEFAULT_LABEL_FIELD_NAME = "label"


def load_multiple_from_file(datasets_params, is_interactive: bool):
    unified_data_dict = dict()
    dataset_dict = read_multiple_to_pandas(datasets_params, is_interactive)

    for dataset_data, dataset_params in zip(dataset_dict.items(), datasets_params):
        dataset_name, file_dict = dataset_data
        dataset_processor = globals().get(f'process_{dataset_params["processor"]}')
        if dataset_processor is not None:
            unified_data_dict[dataset_name] = dataset_processor(file_dict, dataset_params)
        else:
            raise ValueError(f'Unknown dataset processor: {dataset_processor}')
    
    return unified_data_dict

def read_multiple_to_pandas(datasets_params, is_interactive: bool) -> Dict[str, pd.DataFrame]:
    datasets = dict()
    seen = dict()
    for dataset_params in datasets_params:
        filepath, filetype = dataset_params["filepath"], dataset_params["filetype"]
        read_dataset_to_pandas = read_dir_to_pandas if Path(filepath).is_dir() else read_file_to_pandas
        while (df_dict := read_dataset_to_pandas(filepath=filepath, filetype=filetype, is_interactive=is_interactive, config=dataset_params)) is None:
            if is_interactive:
                filetype = input(f'Unknown dataset file format for: {filepath}. Please, input the correct file format manually.\n'
                                f'Options: {", ".join(SUPPORTED_FILETYPES)}\n')
                continue
            raise ValueError(f'Unknown dataset file format for: {filepath}')
        
        name = Path(filepath).stem if filetype != "huggingface" else filepath
        datasets[_get_unique_name(name, seen)] = df_dict
    
    return datasets

def read_dir_to_pandas(filepath: str, filetype: str, is_interactive: bool, config):
    data = dict()
    pathlist = Path(filepath).iterdir()
    for path in pathlist:
        while (df_dict := read_file_to_pandas(str(path), config, filetype=filetype, is_interactive=is_interactive)) is None:
            if is_interactive:
                filetype = input(f'Unknown dataset file format for: {filepath}. Please, input the correct file format manually.\n'
                                f'Options: {", ".join(SUPPORTED_FILETYPES)}\n')
                continue
            raise ValueError(f'Unknown dataset file format for: {filepath}')
        data.update(df_dict)
    
    return data
    

def read_file_to_pandas(filepath: str, config, is_interactive: bool, filetype="auto"):
    data = dict()
    if filetype == "auto":
        filetype = filepath.rsplit(".", 1)[-1]
        
    match filetype:
        case "csv":
            data[Path(filepath).stem] = pd.read_csv(filepath, encoding='utf-8')            
        case "tsv":
            data[Path(filepath).stem] = pd.read_csv(filepath, sep='\t',  encoding='utf-8')
        case "xls" | "xlsx":
            data[Path(filepath).stem] = pd.read_excel(filepath,  encoding='utf-8')
        case "json":
            data[Path(filepath).stem] = pd.read_json(filepath,  encoding='utf-8')
        case "jsonl":
            data[Path(filepath).stem] = pd.read_json(filepath, lines=True,  encoding='utf-8')
        case "xml":
            data[Path(filepath).stem] = pd.read_xml(filepath, encoding='utf-8')
        case "huggingfacehub":
            data[Path(filepath).stem] = datasets.load_dataset(filepath, config["configuration"])
        case _:
            return None

    return data 

#######################################################
#                     PROCESSORS                      #
#######################################################

def process_default(data: Dict[str, pd.DataFrame], config) -> pd.DataFrame:
    if config["filetype"] == "huggingfacehub":
        return process_huggingfacehub(data, config)
    
    dataset_name = list(data.keys())[0]
    data = list(data.values())[0]    
    
    if config["text_field"] not in data.columns or config["label_field"] not in data.columns:
        raise ValueError(f"Could not parse dataset {dataset_name}. "
                         "Text and label fields are not specified correctly. "
                         "Please, correctly specify dataset specifications or " 
                         "define your own custom function for parsing.")
    
    #data = data.loc[:, [config["text_field"], config["label_field"]]]
    data.rename(columns={config["text_field"]: DEFAULT_TEXT_FIELD_NAME}, inplace=True)
    data[DEFAULT_LABEL_FIELD_NAME] = data[config["label_field"]].astype(str) != config["human_label"]
    
    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": data.to_dict(orient="list")}
    
    # Try to stratify, if possible
    try:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=data["label"])
    except ValueError:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=None)
    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}


def process_huggingfacehub(data, config):
    HUMAN_LABEL = 0
    MACHINE_LABEL = 1
    dataset_name = list(data.keys())[0]
    data = list(data.values())[0]

    if isinstance(data, datasets.DatasetDict) and len(data.column_names.keys()) == 1:
        data = data[list(data.column_names.keys())[0]]
    
    if isinstance(data, datasets.DatasetDict) and \
       ((config["train_split"] is not None and \
         config["train_split"] not in data.column_names.keys()) or \
        config["test_split"] not in data.column_names.keys()):
           raise ValueError(f"Could not parse dataset. Incorrectly specified splits for {dataset_name} dataset.")
    
    if isinstance(data, datasets.DatasetDict) and \
       (config["train_split"] is not None and \
        ((config["text_field"] not in data.column_names[config["train_split"]] or \
          (config["label_field"] not in data.column_names[config["train_split"]] and \
           config["processor"] != "human_only" and \
           config["processor"] != "machine_only")))):
           raise ValueError(f"Could not parse dataset. Incorrectly specified text or label fields for {dataset_name} dataset in train split.")

    if isinstance(data, datasets.DatasetDict) and \
        (config["text_field"] not in data.column_names[config["test_split"]] or \
        (config["label_field"] not in data.column_names[config["test_split"]] and \
         config["processor"] != "human_only" and \
         config["processor"] != "machine_only")):
           raise ValueError(f"Could not parse dataset. Incorrectly specified text or label fields for {dataset_name} dataset in test split.")

    if isinstance(data, datasets.Dataset) and \
       (config["text_field"] not in data.column_names or \
        (config["label_field"] not in data.column_names and \
         config["processor"] != "human_only" and \
         config["processor"] != "machine_only")):
           raise ValueError(f"Could not parse dataset. Incorrectly specified text or label fields for {dataset_name} dataset.")
        
    if config["processor"] == "human_only":
        data = _add_labels_to_text_only_huggingface(data, [config["train_split"], config["test_split"]], HUMAN_LABEL)
    if config["processor"] == "machine_only":
        data = _add_labels_to_text_only_huggingface(data, [config["train_split"], config["test_split"]], MACHINE_LABEL)
    
    if config["test_size"] == 1 or config["train_split"] is None:
        return {"train": {"text": [], "label": []}, "test": data[config["test_split"]].to_dict()}
    
    if isinstance(data, datasets.Dataset):
        data = data.to_pandas()
        #data = data.loc[:, [config["text_field"], config["label_field"]]]
        data[DEFAULT_LABEL_FIELD_NAME] = data[config["label_field"]].astype(str) != config["human_label"]
        data.rename(columns={config["text_field"]: DEFAULT_TEXT_FIELD_NAME}, inplace=True)
        # Try to stratify, if possible
        try:
            data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=data["label"])
        except ValueError:
            print(f"Could not stratify train-test split for dataset {config['filepath']}. Continuing without...")
            data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=None)
        
        return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}

        
    return {"train": data[config["train_split"]].to_dict(), "test": data[config["test_split"]].to_dict()}

           

def process_human_only(data, config):
    if config["filetype"] == "huggingfacehub":
        return process_huggingfacehub(data, config)
    
    HUMAN_LABEL = 0
    dataset_name = list(data.keys())[0]
    data = list(data.values())[0]
    
    if len(data.columns) == 1:
        data.columns = [config["text_field"]]
    if config["text_field"] not in data.columns:
        raise ValueError(f"Could not parse dataset {dataset_name}."
                         "Text field is not specified correctly."
                         "Please, correctly specify dataset specifications or" 
                         "define your own custom function for parsing.")
    
    #data = data.loc[:, [config["text_field"]]]
    data[config["label_field"]] = pd.Series([HUMAN_LABEL]*data[config["text_field"]].size)
    
    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": data.to_dict(orient="list")}
    
    # Try to stratify, if possible
    try:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=data["label"])
    except ValueError:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=None)

    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}

    

def process_machine_only(data, config):
    if config["filetype"] == "huggingfacehub":
        return process_huggingfacehub(data, config)
    
    MACHINE_LABEL = 1
    dataset_name = list(data.keys())[0]
    data = list(data.values())[0]
    
    if len(data.columns) == 1:
        data.columns = [config["text_field"]]
    if config["text_field"] not in data.columns:
        raise ValueError(f"Could not parse dataset {dataset_name}."
                         "Text field is not specified correctly."
                         "Please, correctly specify dataset specifications or" 
                         "define your own custom function for parsing.")
    
    #data = data.loc[:, [config["text_field"]]]
    data[config["label_field"]] = pd.Series([MACHINE_LABEL]*data[config["text_field"]].size)
    
    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": data.to_dict(orient="list")}
    
    # Try to stratify, if possible
    try:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=data["label"])
    except ValueError:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=None)

    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}



def process_test_small_dir(data: Dict[str, pd.DataFrame], config):
    human_texts = data["human"]["human"]
    machine_texts = data["machine"]["machine"]
    df_human_labeled = pd.concat([human_texts, 
                                  pd.Series([0]*human_texts.size)], 
                                 axis="columns", 
                                 ignore_index=True, 
                                 names=["text", "label"])
    df_chat_labeled = pd.concat([machine_texts, 
                                 pd.Series([1]*machine_texts.size)], 
                                axis="columns", 
                                ignore_index=True, 
                                names=["text", "label"])

    data = pd.concat([df_human_labeled, df_chat_labeled], ignore_index=True)
    data.columns = ["text", "label"]

    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": data.to_dict(orient="list")}

    # Try to stratify, if possible
    try:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], 
                                                 random_state=42, shuffle=config["shuffle"], 
                                                 stratify=data["label"])
    except ValueError:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], 
                                                 random_state=42, shuffle=config["shuffle"], 
                                                 stratify=None)

    return {"train": data_train.reset_index().to_dict(orient='list'), 
            "test": data_test.reset_index().to_dict(orient='list')}
                  
 
def process_train_test_in_multiple_files(data: Dict[str, pd.DataFrame], config):
    train = data[config["train_split"]]
    test = data[config["test_split"]]
    
    if config["text_field"] not in train.columns and \
       config["text_field"] not in test.columns or \
       config["label_field"] not in train.columns and \
       config["label_field"] not in test.columns:
        raise ValueError(f"Could not parse dataset {config['filepath']}."
                        "Text and label fields are not specified correctly."
                        "Please, correctly specify dataset specifications or" 
                        "define your own custom function for parsing.")

    for subset in [train, test]:
        #subset = subset.loc[:, [config["text_field"], config["label_field"]]]
        subset.rename(columns={config["text_field"]: DEFAULT_TEXT_FIELD_NAME}, inplace=True)
        subset[DEFAULT_LABEL_FIELD_NAME] = subset[config["label_field"]].astype(str) != config["human_label"]
    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": test.to_dict(orient="list")}
    
    return {"train": train.to_dict(orient='list'), "test": test.to_dict(orient='list')}

    

def process_TruthfulQA(data: Dict[str, pd.DataFrame], config) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = config["dataset_other"]
    num_samples = config.get("num_samples", -1)

    a_human = data['Best Answer'].fillna("").where(1 < data['Best Answer'].fillna("").str.split().apply(len)).astype(str)
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(1 < data[f'{detectLLM}_answer'].fillna("").str.split().apply(len)).astype(str)

    df_human_labeled = pd.concat([a_human, pd.Series([0]*a_human.size)], axis="columns", ignore_index=True, names=["text", "label"]).head(floor(num_samples/2))
    df_chat_labeled = pd.concat([a_chat, pd.Series([1]*a_chat.size)], axis="columns", ignore_index=True, names=["text", "label"]).head(floor(num_samples/2))
    
    data = pd.concat([df_human_labeled, df_chat_labeled], ignore_index=True)
    data.columns = ["text", "label"]
    
    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": data.to_dict(orient="list")}
    
    # Try to stratify, if possible
    try:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=data["label"])
    except ValueError:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=None)

    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}


def process_SQuAD1(data: Dict[str, pd.DataFrame], config) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = config["dataset_other"]
    a_human = [eval(ans)['text'][0] for ans in data['answers'].tolist() if len(eval(ans)['text'][0].split()) > 1]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(data[f'{detectLLM}_answer'].str.split().apply(len) > 1)
    
    df_human_labeled = pd.concat([a_human, pd.Series([0]*a_human.size)], axis="columns", ignore_index=True, names=["text", "label"])
    df_chat_labeled = pd.concat([a_chat, pd.Series([1]*a_chat.size)], axis="columns", ignore_index=True, names=["text", "label"])
    
    data = pd.concat([df_human_labeled, df_chat_labeled], ignore_index=True)
    data.columns = ["text", "label"]
    
    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": data.to_dict(orient="list")}
    
    # Try to stratify, if possible
    try:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=data["label"])
    except ValueError:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=None)

    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}



def process_NarrativeQA(data: Dict[str, pd.DataFrame], config) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = config["dataset_other"]
    a_human = [ans.split(";")[0] for ans in data['answers'].tolist() if 1 < len(ans.split(";")[0].split()) < 150]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where((1 < data[f'{detectLLM}_answer'].str.split().apply(len)) & (data[f'{detectLLM}_answer'].str.split().apply(len) < 150))
    
    df_human_labeled = pd.concat([a_human, pd.Series([0]*a_human.size)], axis="columns", ignore_index=True, names=["text", "label"])
    df_chat_labeled = pd.concat([a_chat, pd.Series([1]*a_chat.size)], axis="columns", ignore_index=True, names=["text", "label"])
    
    data = pd.concat([df_human_labeled, df_chat_labeled], ignore_index=True)
    data.columns = ["text", "label"]
    
    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": data.to_dict(orient="list")}
    
    # Try to stratify, if possible
    try:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=data["label"])
    except ValueError:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=None)

    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}


def process_test_small(data: Dict[str, pd.DataFrame], config) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    human_texts = data[data.label == 0]["text"]
    machine_texts = data[data.label == 1]["text"]
    
    df_human_labeled = pd.concat([human_texts, pd.Series([0]*human_texts.size)], axis="columns", ignore_index=True, names=["text", "label"])
    df_chat_labeled = pd.concat([machine_texts, pd.Series([1]*machine_texts.size)], axis="columns", ignore_index=True, names=["text", "label"])
    
    data = pd.concat([df_human_labeled, df_chat_labeled], ignore_index=True)
    data.columns = ["text", "label"]
    
    if config["test_size"] == 1:
        return {"train": {"text": [], "label": []}, "test": data.to_dict(orient="list")}
    
    # Try to stratify, if possible
    try:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=data["label"])
    except ValueError:
        data_train, data_test = train_test_split(data, test_size=config["test_size"], random_state=42, shuffle=config["shuffle"], stratify=None)

    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}
                  

###########################################################
#                         UTILS                           #
###########################################################

def _process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').resmallplace(
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

def _get_unique_name(name: str, seen: Dict[str, int]) -> str:
    if name not in seen:
        seen[name] = 1
        return name
    else:
        seen[name] += 1
        return name + str(seen[name])
    
def _add_labels_to_text_only_huggingface(dataset: Union[datasets.Dataset, datasets.DatasetDict], 
                                        splits: List[str], 
                                        label: str) -> Union[datasets.Dataset, datasets.DatasetDict]:
    if isinstance(dataset, datasets.Dataset):
        dataset = dataset.add_column("label", [label]*dataset.num_rows)
    elif isinstance(dataset, datasets.DatasetDict):
        for split in splits:
            if split is not None:
                dataset[split] = dataset[split].add_column("label", [label]*dataset[split].num_rows)
    return dataset
