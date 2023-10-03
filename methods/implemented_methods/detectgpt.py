from methods.abstract_methods.pertubation_based_experiment import PertubationBasedExperiment
import numpy as np
import transformers
import re
import torch
import torch.nn.functional as F
import random
import time
from tqdm import tqdm
from methods.utils import get_clf_results, get_ll

class DetectGPT(PertubationBasedExperiment):
    
    def __init__(self, data, config):
        name = self.__class__.__name__
        super().__init__(data, name, config)

    def get_score(self, text, perturbed_texts, base_model, base_tokenizer, DEVICE):
        perturbed_lls = get_lls(perturbed_texts, base_model, base_tokenizer, DEVICE)
        lls_std = np.std(perturbed_lls) if len(perturbed_lls) > 1 else 1
        
        if lls_std == 0:
            lls_std = 1
            print("WARNING: std of perturbed original is 0, setting to 1")
            print(
                f"Number of unique perturbed original texts: {len(set(text))}")
            print(f"Original text: {text}")
            
        return np.array([(get_ll(text, base_model, base_tokenizer, DEVICE) - \
                np.mean(get_lls(perturbed_texts, base_model, base_tokenizer, DEVICE))) / \
                lls_std])
                   

def get_lls(texts, model, tokenizer, DEVICE):
    return [get_ll(text, model, tokenizer, DEVICE) for text in texts]