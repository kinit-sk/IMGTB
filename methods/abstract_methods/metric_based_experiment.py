from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from methods.utils import timeit, get_clf_results, load_base_model_and_tokenizer, move_model_to_device
from methods.abstract_methods.experiment import Experiment

class MetricBasedExperiment(Experiment):
    
    def __init__(self, data, name, config):
        super().__init__(data, name)
        self.clf_algorithm_name = config["clf_algo_for_threshold"]
        self.cache_dir = config["cache_dir"]
        self.base_model_name = config["base_model_name"]
        self.DEVICE = config["DEVICE"]
        self.base_model = None
        self.base_tokenizer = None
    
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

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print(f"Using cache dir {self.cache_dir}")
        
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
        x_train = np.expand_dims(x_train, axis=-1)
        y_train = train_label

        test_text = self.data['test']['text']
        test_label = self.data['test']['label']
        test_criterion = [self.criterion_fn(test_text[idx])
                        for idx in tqdm(range(len(test_text)), desc="Computing metrics on test partition")]
        x_test = np.array(test_criterion)
        x_test = np.expand_dims(x_test, axis=-1)
        y_test = test_label

        train_pred, test_pred, train_pred_prob, test_pred_prob, train_res, test_res = get_clf_results(x_train, y_train, x_test, y_test, clf_model_name=self.clf_algorithm_name)
                
        acc_train, precision_train, recall_train, f1_train, auc_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res

        print(f"{self.name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
        print(f"{self.name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
        
        return {
            'name': f'{self.name}_threshold',
            'input_data': self.data,
            'predictions': {'train': train_pred.tolist(), 'test': test_pred.tolist()},
            'machine_prob': {'train': train_pred_prob, 'test': test_pred_prob},
            'metrics_results': {
                'train': {
                    'acc': acc_train,
                    'precision': precision_train,
                    'recall': recall_train,
                    'f1': f1_train
                },
                'test': {
                    'acc': acc_test,
                    'precision': precision_test,
                    'recall': recall_test,
                    'f1': f1_test
                }
            }
        }