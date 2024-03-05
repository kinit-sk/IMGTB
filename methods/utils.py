import transformers
import re
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time
from functools import wraps
import random
import torch
import torch.nn.functional as F
import traceback

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds\n\n')
        return result
    return timeit_wrapper


# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def select_train_data(data, select_num=-1):
    new_train = {
        'text': [],
        'label': [],
    }
    total_num = len(data['train']['text'])
    index_list = list(range(total_num))
    random.seed(0)
    random.shuffle(index_list)
    if select_num == -1:
        return data
    else:
        for i in range(select_num):
            text = data['train']['text'][index_list[i]]
            label = data['train']['label'][index_list[i]]
            new_train['text'].append(text)
            new_train['label'].append(label)
        data['train'] = new_train

    return data


def filter_test_data(data, max_length=25):
    new_test = {
        'text': [],
        'label': [],
    }
    for i in range(len(data['test']['text'])):
        text = data['test']['text'][i]
        label = data['test']['label'][i]
        if len(text.split()) <= max_length:
            new_test['text'].append(text)
            new_test['label'].append(label)
    data['test'] = new_test
    return data


def move_model_to_device(model, DEVICE):
    DEFAULT_DEVICE = "cpu"
    
    print(f'Moving model {model.__class__.__name__} to {DEVICE}...')
    start = time.time()
    
    try:
        model.to(DEVICE)
    except:
        print(f'Moving to default device {DEFAULT_DEVICE}. Failed to move to {DEVICE} because of the below exception:')
        print(traceback.format_exc())
        model.to(DEFAULT_DEVICE)
    
    print(f'Done ({time.time() - start:.2f}s)')

def move_tensor_to_device(tensor, DEVICE):
    DEFAULT_DEVICE = "cpu"
    
    start = time.time()
    try:
        tensor.to(DEVICE)
    except:
        print(f'Moving to default device {DEFAULT_DEVICE}. Failed to move to {DEVICE} because of the below exception:', file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        tensor.to(DEFAULT_DEVICE)

def cal_metrics(label, pred_label, pred_posteriors):
    if len(set(label)) < 2:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average="weighted")
        recall = recall_score(label, pred_label, average="weighted")
        f1 = f1_score(label, pred_label, average="weighted")
        print("Cannot evaluate AUC metric on data with less than 2 labels. Setting AUC to -1...")
        auc = -1.0
    elif len(set(label)) == 2:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average="weighted")
        recall = recall_score(label, pred_label, average="weighted")
        f1 = f1_score(label, pred_label, average="weighted")
        auc = roc_auc_score(label, pred_posteriors)
    else:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average='weighted')
        recall = recall_score(label, pred_label, average='weighted')
        f1 = f1_score(label, pred_label, average='weighted')
        print("Cannot evaluate AUC metric on data with more than 2 labels. Setting AUC to -1...")
        auc = -1.0

    return acc, precision, recall, f1, auc


def get_ll(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1317

def get_rank(text, model, tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        logits = model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                == labels.unsqueeze(-1)).nonzero()
        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

def get_entropy(text, model, tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        logits = model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

def get_llm_deviation(text, model, tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        logits = model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                == labels.unsqueeze(-1)).nonzero()
        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        
        ranks = torch.log(ranks)
        ranks = torch.square(ranks)

        return ranks.float().mean().item()

def get_s5(text, model, tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        outputs = model(**tokenized, labels=labels)
        logits = outputs.logits[:, :-1]

        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)

        labels = tokenized.input_ids[:, 1:]
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
        ranks, timesteps = matches[:, -1], matches[:, -2]
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"
        ranks = ranks.float() + 1  # convert to 1-indexed rank

        return [-outputs.loss.item(), -neg_entropy.sum(-1).mean().item(), ranks.float().mean().item(), torch.log(ranks).float().mean().item(), torch.square(torch.log(ranks)).float().mean().item()]
