from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from methods.utils import get_entropy

import torch
import torch.nn.functional as F
import numpy as np

class EntropyMetric(MetricBasedExperiment):
    def __init__(self, data, config): # Add new arguments, if needed, e.g. base model, DEVICE
        super().__init__(data, self.__class__.__name__, config)
    
    def criterion_fn(self, text: str):
        return np.array([get_entropy(text, self.base_model, self.base_tokenizer, self.DEVICE)])