from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment

class GenericMetricBasedTemplate(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config) # Set your own name or leave it set to the class name
    
    def criterion_fn(self, text: str):
        """
        Implement this method.
        Method takes an input text and computes a numeric score out of it.

        Args:
            text (str)
            
        Returns a numpy array of numeric criteria (numeric scores)
        """
        raise NotImplementedError("Attempted to call an abstract method.")
