from methods.abstract_methods.pertubation_based_experiment import PertubationBasedExperiment
import numpy as np
import transformers
import torch
import torch.nn.functional as F
from tqdm import tqdm
from methods.utils import get_clf_results, get_rank

class DetectLLM_NPR(PertubationBasedExperiment):
    
    def __init__(self, data, config):
        name = self.__class__.__name__
        super().__init__(data, name, config)

    def get_score(self, text, perturbed_texts, base_model, base_tokenizer, DEVICE):
        perturbed_ranks = get_ranks(perturbed_texts, base_model, base_tokenizer, DEVICE, log=True)
        ranks_std = np.std(perturbed_ranks) if len(perturbed_ranks) > 1 else 1
        
        if ranks_std == 0:
            ranks_std = 1
            print("WARNING: std of perturbed original is 0, setting to 1")
            print(
                f"Number of unique perturbed original texts: {len(set(text))}")
            print(f"Original text: {text}")
        
        return np.array([np.mean(perturbed_ranks) / \
               get_rank(text, base_model, base_tokenizer, DEVICE, log=True) / \
               ranks_std])

def get_ranks(texts, model, tokenizer, DEVICE, log=False):
    return [get_rank(text, model, tokenizer, DEVICE, log) for text in texts]