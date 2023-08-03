import random
import datasets
import tqdm
import pandas as pd
import re
from typing import List, Dict, Union, Tuple
from pathlib import Path
from itertools import zip_longest
from sklearn.model_selection import train_test_split


def process_spaces(text):
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


def process_text_truthfulqa_adv(text):

    if "I am sorry" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    if "as an AI language model" in text or "As an AI language model" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    return text


def data_to_unified(human_texts: pd.Series, machine_texts: pd.Series) \
                -> Dict[str, Dict[str, Union[List[str], List[int]]]]:

    texts = pd.concat([human_texts, machine_texts], axis=0, names=["text"]).reset_index(drop=True)
    labels = pd.Series([0]*len(human_texts) + [1]*len(machine_texts))
    data = pd.DataFrame(data={"text": texts, "label": labels}).reset_index(drop=True)

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True, stratify=data["label"])

    return {"train": data_train.reset_index().to_dict(orient='list'), "test": data_test.reset_index().to_dict(orient='list')}

def process_default(data: Dict[str, pd.DataFrame], text_field, label_field, human_label, **kwargs) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    if text_field not in data.columns or label_field not in data.columns:
        raise ValueError("Could not parse dataset. Please, correctly specify dataset specifications or define your own custom function for parsing.")
    human_texts = data[text_field].where(data[label_field].astype("string") == human_label).dropna().reset_index(drop=True)
    machine_texts = data[text_field].where(data[label_field].astype("string") != human_label).dropna().reset_index(drop=True)
    return human_texts, machine_texts

def process_test_small_dir(data: Dict[str, pd.DataFrame], **kwargs):
    human = data["human"]["human"]
    machine = data["machine"]["machine"]
    return human, machine
    

def process_TruthfulQA(data: Dict[str, pd.DataFrame], other: List[str], **kwargs) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = other[0]
    a_human = data['Best Answer'].fillna("").where(1 < data['Best Answer']).tolist()
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(1 < len(data[f'{detectLLM}_answer'].split())).tolist()
    return a_human, a_chat


def process_SQuAD1(data: Dict[str, pd.DataFrame], other: List[str], **kwargs) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = other[0]
    a_human = [eval(ans)['text'][0] for ans in data['answers'].tolist() if len(eval(ans)['text'][0].split()) > 1]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(data[f'{detectLLM}_answer'].str.split().apply(len) > 1)
    return pd.Series(a_human).head(100), a_chat.head(100)


def process_NarrativeQA(data: Dict[str, pd.DataFrame], other: List[str], **kwargs) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    detectLLM = other[0]
    a_human = data['answers'].tolist()
    a_human = [ans.split(";")[0] for ans in ['answers'].tolist() if 1 < len(ans.split(";")[0].split()) < 150]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(1 < len(data[f'{detectLLM}_answer'].split()) < 150).tolist()
    return a_human, a_chat

def process_test_small(data: Dict[str, pd.DataFrame], **kwargs) -> Tuple[pd.Series, pd.Series]:
    data = list(data.values())[0]
    human_texts = data[data.label == 0]["text"]
    machine_texts = data[data.label == 1]["text"]
    return human_texts, machine_texts                       

def load_multiple_from_file(filepaths: List[str], processors: List[str], **kwargs):
    unified_data_dict = dict()
    dataset_dict = read_multiple_to_pandas(filepaths)
    for dataset, processor in zip_longest(dataset_dict.items(), processors):
        dataset_name, file_dict = dataset
        if processor is None: processor = "default"
        dataset_processor = globals().get(f'process_{processor}')
        if processor is not None:
            human_texts, machine_texts = dataset_processor(file_dict, **kwargs)
            unified_data_dict[dataset_name] = data_to_unified(human_texts, machine_texts)
        else:
            raise ValueError(f'Unknown dataset processor: {processor}')
    
    return unified_data_dict

def read_multiple_to_pandas(filepaths: List[str]) -> List[pd.DataFrame]:
    datasets = dict()
    for filepath in filepaths:
        filetype = "auto"
        read_dataset_to_pandas = read_dir_to_pandas if Path(filepath).is_dir() else read_file_to_pandas
        while (df_dict := read_dataset_to_pandas(filepath, filetype)) is None:
            filetype = input(f'Unknown dataset file format for: {filepath}. Please, input the correct file format manually.\n'
                              'Options: auto, csv, tsv, xls, xlsx, json, jsonl, xml, huggingfacehub\n')
        datasets[Path(filepath).stem] = df_dict
    
    return datasets

def read_dir_to_pandas(filepath: str, *args):
    data = dict()
    pathlist = Path(filepath).iterdir()
    for path in pathlist:
        filetype = "auto"
        while (df_dict := read_file_to_pandas(str(path), filetype)) is None:
            filetype = input(f'Unknown dataset file format for: {filepath}. Please, input the correct file format manually.\n'
                              'Options: auto, csv, tsv, xls, xlsx, json, jsonl, xml, huggingfacehub\n')
        data.update(df_dict)
    
    return data
    

def read_file_to_pandas(filepath: str, filetype="auto"):
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