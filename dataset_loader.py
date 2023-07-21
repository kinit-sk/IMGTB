import random
import datasets
import tqdm
import pandas as pd
import re
from typing import List, Dict, Union, Tuple
from pathlib import Path

# you can add more datasets here and write your own dataset parsing function
DATASETS = ['TruthfulQA_LLMs', 'SQuAD1_LMMs', 'NarrativeQA_LMMs', 'test_small']


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


def process_TruthfulQA(f: pd.DataFrame, detectLLM: str) -> Tuple[List[str], List[str]]:
    # f = pd.read_csv("datasets/TruthfulQA_LLMs.csv")
    a_human = f['Best Answer'].fillna("").where(1 < f['Best Answer']).tolist()
    a_chat = f[f'{detectLLM}_answer'].fillna("").where(1 < len(f[f'{detectLLM}_answer'].split())).tolist()
    return a_human, a_chat


def process_SQuAD1(f: pd.DataFrame, detectLLM: str) -> Tuple[List[str], List[str]]:
    # f = pd.read_csv("datasets/SQuAD1_LLMs.csv")
    a_human = [eval(ans)['text'][0] for ans in f['answers'].tolist() if len(eval(ans)['text'][0].split()) > 1]
    a_chat = f[f'{detectLLM}_answer'].fillna("").where(len(f[f'{detectLLM}_answer'].split()) > 1).tolist()
    return a_human, a_chat


def process_NarrativeQA(f: pd.DataFrame, detectLLM: str) -> Tuple[List[str], List[str]]:
    # f = pd.read_csv("datasets/NarrativeQA_LLMs.csv")
    a_human = f['answers'].tolist()
    a_human = [ans.split(";")[0] for ans in ['answers'].tolist() if 1 < len(ans.split(";")[0].split()) < 150]
    a_chat = f[f'{detectLLM}_answer'].fillna("").where(1 < len(f[f'{detectLLM}_answer'].split()) < 150).tolist()
    return a_human, a_chat

def process_test_small(f: pd.DataFrame, **kwargs) -> Tuple[List[str], List[str]]:
    human_texts = f[f.label == 0]["text"]
    machine_texts = f[f.label == 1]["text"]
    return human_texts, machine_texts

def read_file_to_pandas(filepath: str, filetype: str):
    if filetype == "auto":
        filetype = filepath.rsplit(".", 1)[-1]
        print(filepath.rsplit(".", 1))
    match filetype:
        case "csv":
            return pd.read_csv(filepath)
        case "xls" | "xlsx":
            return pd.read_excel(filepath)
        case "json":
            return pd.read_json(filepath)
        case "xml":
            return pd.read_xml(filepath)
        case "huggingface":
            return datasets.load_dataset(filepath).to_pandas()
        case _:
            raise ValueError(f'Unknown dataset file format: {filetype}')

def load_from_file(filepath: str, filetype: str, **kwargs):
    df = read_file_to_pandas(filepath, filetype)
    name = Path(filepath).stem
    
    if name in DATASETS:
        process_dataset = globals()[f'process_{name}']
        human_texts, machine_texts = process_dataset(df, **kwargs)
        return data_to_unified(human_texts, machine_texts)
    
    raise ValueError(f'Unknown dataset: {name}')