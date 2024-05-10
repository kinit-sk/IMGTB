from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from methods.utils import get_ll, get_rank

import numpy as np

class DetectLLM_LLR(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config) # Set your own name or leave it set to the class name
        self.config = config
    
    def criterion_fn(self, text: str):
       ll = np.nan_to_num(get_ll(text, self.base_model, self.base_tokenizer, self.DEVICE))
       log_rank = np.nan_to_num(get_rank(text, self.base_model, self.base_tokenizer, self.DEVICE, log=True))
       return np.array([np.divide(ll, log_rank, out=np.zeros_like(ll), where=log_rank!=0.0)])
