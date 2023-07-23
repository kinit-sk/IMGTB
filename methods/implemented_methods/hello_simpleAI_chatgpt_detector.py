from methods.abstract_methods.supervised_experiment import SupervisedExperiment

class HelloSimpleAIChatGPTDetector(SupervisedExperiment):
     def __init__(self, data, cache_dir, batch_size, DEVICE, **kwargs): # Add extra parameters if needed
        name = self.__class__.__name__
        model = 'Hello-SimpleAI/chatgpt-detector-roberta'
        super().__init__(data, name, model, cache_dir, batch_size, DEVICE, pos_bit=1)

