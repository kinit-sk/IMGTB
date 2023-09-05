from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
import torch
import numpy as np

class LoglikelihoodMetric(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config)
    
    def criterion_fn(self, text: str):
        with torch.no_grad():
            tokenized = self.base_tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.DEVICE)
            labels = tokenized.input_ids
            return np.array([-self.base_model(**tokenized, labels=labels).loss.item()])
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1317