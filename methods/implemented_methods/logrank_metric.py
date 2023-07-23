from methods.implemented_methods.rank_metric import RankMetric
import torch

class LogRankMetric(RankMetric):
    def __init__(self, data, model, tokenizer, DEVICE, log=False, **kwargs):
        super().__init__(data, model, tokenizer, DEVICE, log=True)