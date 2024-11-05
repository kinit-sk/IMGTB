import os
import time
import gc

import transformers
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from methods.utils import timeit, move_model_to_device, cal_metrics
from methods.abstract_methods.experiment import Experiment

class AUCThresholdCalibrator:
    def __init__(self):
        self.threshold = None

    def fit(self, metrics, labels):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, metrics)
        th_optim_idx = np.argmax(tpr - fpr)
        th_optim2_idx = np.argmin(np.abs(fpr+tpr-1))
        if fpr[th_optim_idx] < fpr[th_optim2_idx]:
            self.threshold = thresholds[th_optim_idx]
        else:
            self.threshold = thresholds[th_optim2_idx]
        print(self.threshold)
        return self
    
    def predict(self, x):
        if self.threshold is None:
            raise ValueError("Cannot predict without specified threshold. Please, run the fit() method to estimate a threshold first.")
        
        return [0 if sample < self.threshold else 1 for sample in x]

    def predict_proba(self, x):
        return [(1, 0) if pred==0 else (0, 1) for pred in self.predict(x)]

class ManualThresholdSelectionClassifier:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def fit(self, x, y):
        print(self.threshold)
        return self

    def predict(self, x):
        if self.threshold is None:
            raise ValueError("Cannot predict without specified threshold. Please, specify the threshold parameter in the configurations manually.")
        
        return [0 if sample < self.threshold else 1 for sample in x]

    def predict_proba(self, x):
        return [(1, 0) if pred==0 else (0, 1) for pred in self.predict(x)]


THRESHOLD_ESTIMATION_MODELS = {
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "MLPClassifier": MLPClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "AUCThresholdCalibrator": AUCThresholdCalibrator,
    "ManualThresholdSelectionClassifier":  ManualThresholdSelectionClassifier
}

DEFAULT_THRESHOLD_ESTIMATION_MODEL = "AUCThresholdCalibrator"
DEFAULT_TEST_ONLY_THRESHOLD_ESTIMATION_MODEL = "ManualThresholdSelectionClassifier"

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
        # Moving to a new parameter name for threshold estimation algorithm, but leaving also the old one as backup for backwards compatibility
        self.threshold_estimation_params = params if (params:=config.get("threshold_estimation_params")) is not None else config.get("clf_algo_for_threshold")
        self.threshold_estimation_params =  self.threshold_estimation_params if  self.threshold_estimation_params is not None else {}
        self.threshold_estimation_params["name"] = name if (name:=self.threshold_estimation_params.get("name")) is not None else DEFAULT_THRESHOLD_ESTIMATION_MODEL

    
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
        self.base_model, self.base_tokenizer = self.load_base_model_and_tokenizer(
            self.base_model_name, self.cache_dir)
        move_model_to_device(self.base_model, self.DEVICE)
            
        torch.manual_seed(0)
        np.random.seed(0)

        # get train data
        train_text = self.data['train']['text']
        train_label = self.data['train']['label']
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
        train_pred, test_pred, train_pred_prob, test_pred_prob, train_res, test_res = self.get_clf_results(x_train, y_train, x_test, y_test)
                
        acc_train, precision_train, recall_train, f1_train, auc_train, specificity_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test, specificity_test = test_res

        print(f"{self.name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}, specificity_train: {specificity_train}")
        print(f"{self.name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}, specificity_test: {specificity_test}")
        
        end_time = time.time()
        
         # Clean up
        del self.base_model
        gc.collect()
        torch.cuda.empty_cache()

        
        return {
            'name': f'{self.name}_threshold',
            'type': 'metric-based',
            'input_data': self.data,
            'predictions': {'train': train_pred, 'test': test_pred},
            'machine_prob': {'train': train_pred_prob, 'test': test_pred_prob},
            'criterion': {'train': [elem.tolist() for elem in train_criterion], 'test': [elem.tolist() for elem in test_criterion]},
            'running_time_seconds': end_time - start_time,
            'metrics_results': {
                'train': {
                    'acc': acc_train,
                    'precision': precision_train,
                    'recall': recall_train,
                    'f1': f1_train,
                    'specificity': specificity_train
                },
                'test': {
                    'acc': acc_test,
                    'precision': precision_test,
                    'recall': recall_test,
                    'f1': f1_test,
                    'specificity': specificity_test
                }
            },
            "config": self.config
        }

    def get_clf_results(self, x_train, y_train, x_test, y_test):
        model_name = self.threshold_estimation_params["name"]
        model = THRESHOLD_ESTIMATION_MODELS[model_name]
        if len(x_train) == 0 or self.threshold is not None:
             model_instance = THRESHOLD_ESTIMATION_MODELS.get(DEFAULT_TEST_ONLY_THRESHOLD_ESTIMATION_MODEL)(self.threshold)
        else:
             model_instance = model(**{key: value for key, value in self.threshold_estimation_params.items() if key != 'name'})
        clf = model_instance.fit(x_train, y_train)

        y_train_pred = clf.predict(x_train)
        y_train_pred_prob = clf.predict_proba(x_train)
        y_train_pred_prob = [_[1] for _ in y_train_pred_prob]
        acc_train, precision_train, recall_train, f1_train, auc_train, specificity_train = cal_metrics(
            y_train, y_train_pred, y_train_pred_prob)
        train_res = acc_train, precision_train, recall_train, f1_train, auc_train, specificity_train

        y_test_pred = clf.predict(x_test)
        y_test_pred_prob = clf.predict_proba(x_test)
        
        y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
        acc_test, precision_test, recall_test, f1_test, auc_test, specificity_test = cal_metrics(
            y_test, y_test_pred, y_test_pred_prob)
        test_res = acc_test, precision_test, recall_test, f1_test, auc_test, specificity_test

        return y_train_pred, y_test_pred, y_train_pred_prob, y_test_pred_prob, train_res, test_res

    def load_base_model_and_tokenizer(self, name, cache_dir):

        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            name, cache_dir=cache_dir)
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            name, cache_dir=cache_dir)
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if base_tokenizer.pad_token is None:
            base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            base_model.resize_token_embeddings(len(base_tokenizer))


        return base_model, base_tokenizer