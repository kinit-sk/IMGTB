import os
import time

import torch
import numpy as np


from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment

class GLTRMetric(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config)
        self.base_model_name = config["base_model_name"]
        self.config = config
        if self.threshold_estimation_params.get("name") == "AUCThresholdCalibrator" or self.threshold_estimation_params.get("name") == "AUCThresholdCalibrator" == "ManualThresholdSelectionClassifier":
            raise ValueError("GLTRMetric is not compatible with threshold estimation using AUCThresholdCalibrator or ManualThresholdSelectionClassifier. Please, manually specify a different classfier capable of classifying multi-dimensional sample data points.")
    
    def criterion_fn(self, text: str):
        with torch.no_grad():
            tokenized = self.base_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.DEVICE)
            logits = self.base_model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]

            # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True)
                    == labels.unsqueeze(-1)).nonzero()

            assert matches.shape[
                1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"
            ranks = ranks.float()
            res = np.array([0.0, 0.0, 0.0, 0.0])
            for i in range(len(ranks)):
                if ranks[i] < 10:
                    res[0] += 1
                elif ranks[i] < 100:
                    res[1] += 1
                elif ranks[i] < 1000:
                    res[2] += 1
                else:
                    res[3] += 1
            if res.sum() > 0:
                res = res / res.sum()

            return res
