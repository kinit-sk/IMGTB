from abstract_methods.metric_based_experiment import MetricBasedExperiment

class GenericMetricBasedTemplate(MetricBasedExperiment):
    def __init__(self, data, **kwargs): # Add new arguments, if needed, e.g. base model, DEVICE
        super().__init__(data, self.__class__.__name__) # Set your own name or leave it set to the class name
    
    def criterion_fn(self, text: str):
        """
        Implement this method.
        Method takes an input text and computes a numeric score out of it.

        Args:
            text (str)
            
        Returns a numeric score assigned to the input text by the criterion.
        """
        raise NotImplementedError("Attempted to call an abstract method.")
