from methods.abstract_methods.pertubation_based_experiment import PertubationBasedExperiment
from methods.utils import get_rank, get_ll, get_entropy, get_llm_deviation
import numpy as np

class PerturbationMFD(PertubationBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config)
    
    def get_score(self, text, perturbed_texts, base_model, base_tokenizer, DEVICE):
        perturbed_ranks = get_multiple(get_rank, perturbed_texts, base_model, base_tokenizer, DEVICE, log=True)
        perturbed_lls = get_multiple(get_ll, perturbed_texts, base_model, base_tokenizer, DEVICE)
        perturbed_entropies = get_multiple(get_entropy, perturbed_texts, base_model, base_tokenizer, DEVICE)
        perturbed_llm_dev = get_multiple(get_llm_deviation, perturbed_texts, base_model, base_tokenizer, DEVICE)
        
        pert_rank_std = np.std(perturbed_ranks) if np.std(perturbed_ranks) != 0 else 1
        pert_ll_std = np.std(perturbed_lls) if np.std(perturbed_lls) != 0 else 1
        pert_entr_std = np.std(perturbed_entropies) if np.std(perturbed_entropies) != 0 else 1
        pert_llm_dev_std = np.std(perturbed_llm_dev) if np.std(perturbed_llm_dev) != 0 else 1
        
        return np.array([get_rank(text, self.base_model, self.base_tokenizer, self.DEVICE, log=True),
                get_ll(text, self.base_model, self.base_tokenizer, self.DEVICE),
                get_entropy(text, self.base_model, self.base_tokenizer, self.DEVICE),
                get_llm_deviation(text, self.base_model, self.base_tokenizer, self.DEVICE),
                np.mean(perturbed_ranks) / pert_rank_std,
                np.mean(perturbed_lls) / pert_ll_std,
                np.mean(perturbed_entropies) / pert_entr_std,
                np.mean(perturbed_llm_dev) / pert_llm_dev_std])

def get_multiple(function, texts, base_model, base_tokenizer, DEVICE, **kwargs):
    return [function(text, base_model, base_tokenizer, DEVICE, **kwargs) for text in texts]