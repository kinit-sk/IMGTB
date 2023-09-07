import transformers
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import time
from functools import wraps
import random
import torch
import torch.nn.functional as F

CLF_MODELS = {
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "MLPClassifier": MLPClassifier,
    "AdaBoostClassifier": AdaBoostClassifier
}

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


def load_base_model_and_tokenizer(name, cache_dir):

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    if base_tokenizer.pad_token is None:
        base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_model.resize_token_embeddings(len(base_tokenizer))


    return base_model, base_tokenizer


def move_model_to_device(base_model, DEVICE):
    print(f'MOVING BASE MODEL TO {DEVICE}...', end='', flush=True)
    start = time.time()

    base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def cal_metrics(label, pred_label, pred_posteriors):
    if len(set(label)) < 2:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        print("Cannot evaluate AUC metric on data with less than 2 labels. Setting AUC to 0...")
        auc = 0
    elif len(set(label)) == 2:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_posteriors)
    else:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average='weighted')
        recall = recall_score(label, pred_label, average='weighted')
        f1 = f1_score(label, pred_label, average='weighted')
        auc = -1.0
        conf_m = confusion_matrix(label, pred_label)
        print(conf_m)
    return acc, precision, recall, f1, auc


def get_clf_results(x_train, y_train, x_test, y_test, config):

    clf_algo_config = config["clf_algo_for_threshold"]
    clf_algo_name = clf_algo_config["name"]
    if CLF_MODELS.get(clf_algo_name) is None:
        raise ValueError(f"Unsupported classification algorithm for threshold computation selected: {clf_algo_name}")
    
    clf_algo = CLF_MODELS[clf_algo_name]
    clf_model = clf_algo(**{key: value for key, value in clf_algo_config.items() if key != 'name'})
    clf = clf_model.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_train_pred_prob = clf.predict_proba(x_train)
    y_train_pred_prob = [_[1] for _ in y_train_pred_prob]
    acc_train, precision_train, recall_train, f1_train, auc_train = cal_metrics(
        y_train, y_train_pred, y_train_pred_prob)
    train_res = acc_train, precision_train, recall_train, f1_train, auc_train

    y_test_pred = clf.predict(x_test)
    y_test_pred_prob = clf.predict_proba(x_test)
    y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
    acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
        y_test, y_test_pred, y_test_pred_prob)
    test_res = acc_test, precision_test, recall_test, f1_test, auc_test

    return y_train_pred, y_test_pred, y_train_pred_prob, y_test_pred_prob, train_res, test_res

def get_ll(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1317

def get_rank(text, model, tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(DEVICE)
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
        tokenized = tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

def get_llm_deviation(text, model, tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(DEVICE)
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
