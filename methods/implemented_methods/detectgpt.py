from methods.abstract_methods.pertubation_based_experiment import PertubationBasedExperiment
import numpy as np
import transformers
import re
import torch
import torch.nn.functional as F
import random
import time
from tqdm import tqdm
from methods.utils import get_clf_results, get_ll, get_lls

class DetectGPT(PertubationBasedExperiment):
    
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_perturbation_results(self, train, test, base_model, base_tokenizer, args):
        
        for res in tqdm(train, desc="Computing log likelihoods"):
            p_ll = get_lls(res["perturbed_text"], base_model,
                        base_tokenizer, args.DEVICE)
            res["ll"] = get_ll(res["text"], base_model,
                            base_tokenizer, args.DEVICE)
            res["all_perturbed_ll"] = p_ll
            res["perturbed_ll_mean"] = np.mean(p_ll)
            res["perturbed_ll_std"] = np.std(p_ll) if len(p_ll) > 1 else 1

        for res in tqdm(test, desc="Computing log likelihoods"):
            p_ll = get_lls(res["perturbed_text"], base_model,
                        base_tokenizer, args.DEVICE)
            res["ll"] = get_ll(res["text"], base_model,
                            base_tokenizer, args.DEVICE)
            res["all_perturbed_ll"] = p_ll
            res["perturbed_ll_mean"] = np.mean(p_ll)
            res["perturbed_ll_std"] = np.std(p_ll) if len(p_ll) > 1 else 1
            
        results = {"train": train, "test": test}
        return results

    def evaluate_perturbation_results(self, args, results, criterion, span_length=10, n_perturbations=1):
        # Train
        train_predictions = []
        for res in results['train']:
            if criterion == 'd':
                train_predictions.append(res['ll'] - res['perturbed_ll_mean'])
            elif criterion == 'z':
                if res['perturbed_ll_std'] == 0:
                    res['perturbed_ll_std'] = 1
                    print("WARNING: std of perturbed original is 0, setting to 1")
                    print(
                        f"Number of unique perturbed original texts: {len(set(res['perturbed_text']))}")
                    print(f"Original text: {res['text']}")

                train_predictions.append(
                    (res['ll'] - res['perturbed_ll_mean']) / res['perturbed_ll_std'])

        # Test
        test_predictions = []
        for res in results['test']:
            if criterion == 'd':
                test_predictions.append(res['ll'] - res['perturbed_ll_mean'])
            elif criterion == 'z':
                if res['perturbed_ll_std'] == 0:
                    res['perturbed_ll_std'] = 1
                    print("WARNING: std of perturbed original is 0, setting to 1")
                    print(
                        f"Number of unique perturbed original texts: {len(set(res['perturbed_text']))}")
                    print(f"Original text: {res['text']}")

                test_predictions.append(
                    (res['ll'] - res['perturbed_ll_mean']) / res['perturbed_ll_std'])

        x_train = train_predictions
        x_train = np.expand_dims(x_train, axis=-1)
        y_train = [_['label'] for _ in results['train']]

        x_test = test_predictions
        x_test = np.expand_dims(x_test, axis=-1)
        y_test = [_['label'] for _ in results['test']]

        name = f'perturbation_{n_perturbations}_{criterion}'

        train_pred, test_pred, train_pred_prob, test_pred_prob, train_res, test_res = get_clf_results(x_train, y_train, x_test, y_test)
        acc_train, precision_train, recall_train, f1_train, auc_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res

        print(f"{name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
        print(f"{name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

        return {
            'name': name,
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
            },
            'perturbations_info': {
                'pct_words_masked': args.pct_words_masked,
                'span_length': span_length,
                'n_perturbations': n_perturbations,
            },
            'raw_results': results
        }
        