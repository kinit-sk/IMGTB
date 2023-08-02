from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from methods.utils import get_rank
import torch


class LogRankMetric(MetricBasedExperiment):
    def __init__(self, data, model, tokenizer, DEVICE, **kwargs): # Add new arguments, if needed, e.g. base model, DEVICE
        super().__init__(data, self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.DEVICE = DEVICE
    
    def criterion_fn(self, text: str):
        return get_rank(text, self.model, self.tokenizer, self.DEVICE, log=True)