from methods.abstract_methods.supervised_experiment import SupervisedExperiment

class DistilBertBaseUncased(SupervisedExperiment):
     def __init__(self, data, config): # Add extra parameters if needed
        name = self.__class__.__name__
        model = 'distilbert-base-uncased'
        super().__init__(data, name, model, config, finetune=True, pos_bit=1)

