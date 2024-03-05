from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from methods.utils import timeit, move_model_to_device, cal_metrics
from methods.abstract_methods.experiment import Experiment
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


CLF_MODELS = {
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "MLPClassifier": MLPClassifier,
    "AdaBoostClassifier": AdaBoostClassifier
}


class MetricBasedExperiment(Experiment):
    
    def __init__(self, data, name, config):
        super().__init__(data, name)
        self.cache_dir = config["cache_dir"]
        self.base_model_name = config["base_model_name"]
        self.DEVICE = config["DEVICE"]
        self.base_model = None
        self.base_tokenizer = None
        self.threshold = config.get("threshold")
        self.config = config
    
    def criterion_fn(self, text: str):
        """
        This is an abstract methods only. You should overwrite it in your own child class.
        Method takes an input text and computes a numeric score out of it.

        Args:
            text (str)
            
        Returns a numeric score assigned to the input text by the criterion.
        """
        raise NotImplementedError("Attempted to call an abstract method.")

    @timeit
    def run(self):
        start_time = time.time()
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print(f"Using cache dir {self.cache_dir}")
        
        if "cuda" in self.DEVICE and not torch.cuda.is_available():
            print(f'Setting default device to cpu. Cuda is not available.')
            self.DEVICE = "cpu"

        print(f"Loading BASE model {self.base_model_name}\n")
        self.base_model, self.base_tokenizer = load_base_model_and_tokenizer(
            self.base_model_name, self.cache_dir)
        move_model_to_device(self.base_model, self.DEVICE)
            
        torch.manual_seed(0)
        np.random.seed(0)

        # get train data
        train_text = self.data['train']['text']
        train_label = self.data['train']['label']
        t1 = time.time()
        train_criterion = [self.criterion_fn(train_text[idx])
                        for idx in tqdm(range(len(train_text)), desc="Computing metrics on train partition")]
        x_train = np.array(train_criterion)
        y_train = train_label

        test_text = self.data['test']['text']
        test_label = self.data['test']['label']
        test_criterion = [self.criterion_fn(test_text[idx])
                        for idx in tqdm(range(len(test_text)), desc="Computing metrics on test partition")]
        x_test = np.array(test_criterion)
        y_test = test_label
        train_pred, test_pred, train_pred_prob, test_pred_prob, train_res, test_res = get_clf_results(x_train, y_train, x_test, y_test, config=self.config)
                
        acc_train, precision_train, recall_train, f1_train, auc_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res

        print(f"{self.name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
        print(f"{self.name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
        
        end_time = time.time()
        
         # Clean up
        del self.base_model
        gc.collect()
        torch.cuda.empty_cache()

        
        return {
            'name': f'{self.name}_threshold',
            'type': 'metric-based',
            'input_data': self.data,
            'predictions': {'train': train_pred.tolist(), 'test': test_pred.tolist()},
            'machine_prob': {'train': train_pred_prob, 'test': test_pred_prob},
            'criterion': {'train': [elem.tolist() for elem in train_criterion], 'test': [elem.tolist() for elem in test_criterion]},
            'running_time_seconds': end_time - start_time,
            'metrics_results': {
                'train': {
                    'acc': acc_train,
                    'precision': precision_train,
                    'recall': recall_train,
                    'f1': f1_train
                },
                'test': {mator based on name
                    'acc': acc_test,
                    'precision': precision_test,
                    'recall': recall_test,
                    'f1': f1_test
                }
            },
            "config": self.config
        }


def threshold_calibration():
    pass

def get_clf_results(x_train, y_train, x_test, y_test, config):

    clf_algo_config = config["clf_algo_for_threshold"]
    clf_algo_name = clf_algo_config["name"]
    if clf_algo_name is None:
        
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
