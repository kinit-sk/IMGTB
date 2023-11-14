from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment

from random import random, seed
import numpy as np

class MiracleMetric(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config)
    
    def criterion_fn(self, text: str):
        seed(42)
        return np.array([random()])
    
    

    

