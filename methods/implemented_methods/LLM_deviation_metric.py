from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from methods.utils import get_llm_deviation
import torch
import numpy as np


class LLMDeviationMetric(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config)
    
    def criterion_fn(self, text: str):
        return np.array([get_llm_deviation(text, self.base_model, self.base_tokenizer, self.DEVICE)])