import random
import datasets
import tqdm
import pandas as pd
import re
from typing import List, Dict, Union, Tuple
from pathlib import Path
from itertools import zip_longest

# you can add more datasets here and write your own dataset parsing function
PROCESSORS = ['TruthfulQA_LLMs', 'SQuAD1_LMMs', 'NarrativeQA_LMMs', 'test_small']


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


def data_to_unified(human_texts: List[str], machine_texts: List[str]) \
                -> Dict[str, Dict[str, Union[List[str], List[int]]]]:
    
    res = list(zip(human_texts, machine_texts))
    
    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][0]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(1)

    return data_new


def process_TruthfulQA(data_list: List[pd.DataFrame], detectLLM: List[str]) -> Tuple[List[str], List[str]]:
    data = data_list[0]
    detectLLM = detectLLM[0]
    a_human = data['Best Answer'].fillna("").where(1 < data['Best Answer']).tolist()
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(1 < len(data[f'{detectLLM}_answer'].split())).tolist()
    return a_human, a_chat


def process_SQuAD1(data_list: List[pd.DataFrame], detectLLM: List[str]) -> Tuple[List[str], List[str]]:
    data = data_list[0]
    detectLLM = detectLLM[0]
    a_human = [eval(ans)['text'][0] for ans in data['answers'].tolist() if len(eval(ans)['text'][0].split()) > 1]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(len(data[f'{detectLLM}_answer'].split()) > 1).tolist()
    return a_human, a_chatf


def process_NarrativeQA(data_list: List[pd.DataFrame], detectLLM: List[str]) -> Tuple[List[str], List[str]]:
    data = data_list[0]
    detectLLM = detectLLM[0]
    a_human = data['answers'].tolist()
    a_human = [ans.split(";")[0] for ans in ['answers'].tolist() if 1 < len(ans.split(";")[0].split()) < 150]
    a_chat = data[f'{detectLLM}_answer'].fillna("").where(1 < len(data[f'{detectLLM}_answer'].split()) < 150).tolist()
    return a_human, a_chat

def process_test_small(data_list: List[pd.DataFrame], *args) -> Tuple[List[str], List[str]]:
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
            case "xls" | "xlsx":
                dfs.append(pd.read_excel(filepath))
            case "json":
                dfs.append(pd.read_json(filepath))
            case "xml":
                dfs.append(pd.read_xml(filepath))
            case "huggingface":
                dfs.append(datasets.load_dataset(filepath).to_pandas())
            case _:
                raise ValueError(f'Unknown dataset file format: {filetype}')
    
    return dfs

def load_from_file(filepaths: List[str], filetypes: List[str], processor_name: str, *args):
    dfs = read_file_to_pandas(filepaths, filetypes)
    dataset_processor = globals().get(f'process_{processor_name}')
    if dataset_processor is not None:
        human_texts, machine_texts = dataset_processor(dfs, *args)
        return data_to_unified(human_texts, machine_texts)
    
    raise ValueError(f'Unknown dataset processor: {processor_name}')