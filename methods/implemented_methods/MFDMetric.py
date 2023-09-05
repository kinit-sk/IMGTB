from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from methods.utils import get_rank, get_ll, get_entropy, get_llm_deviation
import numpy as np

class MFDMetric(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config)
    
    def criterion_fn(self, text: str):
        return np.array([get_rank(text, self.base_model, self.base_tokenizer, self.DEVICE, log=True),
                get_ll(text, self.base_model, self.base_tokenizer, self.DEVICE),
                get_entropy(text, self.base_model, self.base_tokenizer, self.DEVICE),
                get_llm_deviation(text, self.base_model, self.base_tokenizer, self.DEVICE)])