from methods.abstract_methods.supervised_experiment import SupervisedExperiment

class RobertaBaseOpenAIDetector(SupervisedExperiment):
     def __init__(self, data, config): # Add extra parameters if needed
        name = self.__class__.__name__
        model = 'roberta-base-openai-detector'
        super().__init__(data, name, model, config)

