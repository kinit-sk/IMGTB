from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from methods.utils import get_ll, get_rank

import numpy as np

class DetectLLM_LLR(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config) # Set your own name or leave it set to the class name
        self.config = config
    
    def criterion_fn(self, text: str):
       return np.array([get_ll(text, self.base_model, self.base_tokenizer, self.config["DEVICE"]) \
              / get_rank(text, self.base_model, self.base_tokenizer, self.config["DEVICE"], log=True)])

