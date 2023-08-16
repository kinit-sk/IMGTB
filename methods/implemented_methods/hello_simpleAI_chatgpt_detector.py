from methods.abstract_methods.supervised_experiment import SupervisedExperiment

class HelloSimpleAIChatGPTDetector(SupervisedExperiment):
     def __init__(self, data, config): # Add extra parameters if needed
        name = self.__class__.__name__
        model = 'Hello-SimpleAI/chatgpt-detector-roberta'
        super().__init__(data, name, model, config, pos_bit=1)

