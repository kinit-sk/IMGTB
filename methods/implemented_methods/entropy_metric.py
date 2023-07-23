from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
import torch
import torch.nn.functional as F


class EntropyMetric(MetricBasedExperiment):
    def __init__(self, data, model, tokenizer, DEVICE, **kwargs): # Add new arguments, if needed, e.g. base model, DEVICE
        super().__init__(data, self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.DEVICE = DEVICE
    
    def criterion_fn(self, text: str):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.DEVICE)
            logits = self.model(**tokenized).logits[:, :-1]
            neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
            return -neg_entropy.sum(-1).mean().item()