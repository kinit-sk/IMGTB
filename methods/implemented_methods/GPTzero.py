import os
import requests
import time
from tqdm import tqdm
from methods.abstract_methods.experiment import Experiment
from methods.utils import timeit, cal_metrics

# from https://github.com/Haste171/gptzero


class GPTZeroAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.gptzero.me/v2/predict'

    def text_predict(self, document):
        url = f'{self.base_url}/text'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'document': document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def file_predict(self, file_path):
        url = f'{self.base_url}/files'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key
        }
        files = {
            'files': (os.path.basename(file_path), open(file_path, 'rb'))
        }
        response = requests.post(url, headers=headers, files=files)
        return response.json()


class GPTZero(Experiment):
     def __init__(self, data, config): # Add extra parameters if needed
        name = self.__class__.__name__ # Set your own name or leave it set to the class name
        super().__init__(data, name)
        self.gptzero_key = config["gptzero_key"]
        self.config = config
     
     def run(self):
        start_time = time.time()
        
        gptzero_api = GPTZeroAPI(self.gptzero_key)

        train_text = self.data['train']['text']
        train_label = self.data['train']['label']
        test_text = self.data['test']['text']
        test_label = self.data['test']['label']

        train_pred_prob = [gptzero_api.text_predict(
            _)['documents'][0]["completely_generated_prob"] for _ in tqdm(train_text)]
        test_pred_prob = [gptzero_api.text_predict(
            _)['documents'][0]["completely_generated_prob"] for _ in tqdm(test_text)]
        train_pred = [round(_) for _ in train_pred_prob]
        test_pred = [round(_) for _ in test_pred_prob]

        acc_train, precision_train, recall_train, f1_train, auc_train, specificity_train = cal_metrics(
            train_label, train_pred, train_pred_prob)
        acc_test, precision_test, recall_test, f1_test, auc_test, specificity_test = cal_metrics(
            test_label, test_pred, test_pred_prob)

        print(
            f"GPTZero acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}, specificity_train: {specificity_train}")
        print(
            f"GPTZero acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}, specificity_test: {specificity_test}")

        return {
            'name': 'GPTZero',
            'type': 'supervised',
            'predictions': {'train': train_pred, 'test': test_pred},
            'machine_prob': {'train': train_pred_prob, 'test': test_pred_prob},
            'running_time_seconds': time.time() - start_time,
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

