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

def process_default(data_list: List[pd.DataFrame], text_field, label_field, human_label, **kwargs):
    data = data_list[0]
    if text_field not in data.columns or label_field not in data.columns:
        raise ValueError("Could not parse dataset. Please, correctly specify dataset specifications or define your own custom function for parsing.")
    human_texts = data[text_field].where(data[label_field].astype("string") == human_label).dropna().reset_index(drop=True)
    machine_texts = data[text_field].where(data[label_field].astype("string") != human_label).dropna().reset_index(drop=True)
    return human_texts, machine_texts
    

def process_TruthfulQA(data_list: List[pd.DataFrame], other: List[str]) -> Tuple[List[str], List[str]]:
    data = data_list[0]
    detectLLM = other[0]
    a_human = data['Best Answer'].fillna("").where(1 < data['Best Answer']).tolist()
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(1 < len(data[f'{detectLLM}_answer'].split())).tolist()
    return a_human, a_chat


def process_SQuAD1(data_list: List[pd.DataFrame], other: List[str]) -> Tuple[List[str], List[str]]:
    data = data_list[0]
    detectLLM = other[0]
    a_human = [eval(ans)['text'][0] for ans in data['answers'].tolist() if len(eval(ans)['text'][0].split()) > 1]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(len(data[f'{detectLLM}_answer'].split()) > 1).tolist()
    return a_human, a_chat


def process_NarrativeQA(data_list: List[pd.DataFrame], other: List[str]) -> Tuple[List[str], List[str]]:
    data = data_list[0]
    detectLLM = other[0]
    a_human = data['answers'].tolist()
    a_human = [ans.split(";")[0] for ans in ['answers'].tolist() if 1 < len(ans.split(";")[0].split()) < 150]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(1 < len(data[f'{detectLLM}_answer'].split()) < 150).tolist()
    return a_human, a_chat

def process_test_small(data_list: List[pd.DataFrame], **kwargs) -> Tuple[List[str], List[str]]:
    data = data_list[0]
    human_texts = data[data.label == 0]["text"]
    machine_texts = data[data.label == 1]["text"]
    return human_texts, machine_texts

def read_file_to_pandas(filepaths: List[str], filetypes: List[str]) -> List[pd.DataFrame]:
    dfs = []
    for filepath, filetype in zip_longest(filepaths, filetypes):
        
        if filetype is None or filetype == "auto":
            # Try to detect by suffix
            filetype = filepath.rsplit(".", 1)[-1]
        
        match filetype:
            case "csv":
                dfs.append(pd.read_csv(filepath))
            case "tsv":
                dfs.append(pd.read_csv(filepath, sep='\t'))
            case "xls" | "xlsx":
                dfs.append(pd.read_excel(filepath))
            case "json":
                dfs.append(pd.read_json(filepath))#
            case "xml":
                dfs.append(pd.read_xml(filepath))
            case "huggingfacehub":
                dfs.append(datasets.load_dataset(filepath).to_pandas())
            case "zip" | "gz":
                raise ValueError(f'Please, specify a file format'
                                 '(with the "--dataset_filetypes" option)'
                                 'for the compressed file: {filepath}')
            case _:
                raise ValueError(f'Unknown dataset file format: {filetype}')
    
    return dfs

def load_from_file(filepaths: List[str], filetypes: List[str], processor: str, **kwargs):
    dfs = read_file_to_pandas(filepaths, filetypes)
    dataset_processor = globals().get(f'process_{processor}')
    if dataset_processor is not None:
        human_texts, machine_texts = dataset_processor(dfs, **kwargs)
        return data_to_unified(human_texts, machine_texts)
    
    raise ValueError(f'Unknown dataset processor: {processor}')