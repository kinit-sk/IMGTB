from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from methods.utils import get_ll, get_rank

class DetectLLM_LLR(MetricBasedExperiment):
    def __init__(self, data, model, tokenizer, args, **kwargs):
        super().__init__(data, self.__class__.__name__) # Set your own name or leave it set to the class name
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
    
    def criterion_fn(self, text: str):
       return get_ll(text, self.model, self.tokenizer, self.args.DEVICE) \
              / get_rank(text, self.model, self.tokenizer, self.args.DEVICE, log=True)

