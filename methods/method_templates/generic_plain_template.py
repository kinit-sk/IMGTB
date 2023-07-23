from methods.abstract_methods.supervised_experiment import SupervisedExperiment

class GenericSupervisedTemplate(SupervisedExperiment):
     def __init__(self, data, **kwargs): # Add extra parameters if needed
        name = self.__class__.__name__ # Set your own name or leave it set to the class name
        # Define all parameters for the instantiation of SupervisedExperiment, e.g. model name, DEVICE, etc.
        # super().__init__(data, name, model, cache_dir, batch_size, DEVICE...)
        pass

