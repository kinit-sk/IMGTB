from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
import torch

class RankMetric(MetricBasedExperiment):
    def __init__(self, data, model, tokenizer, DEVICE, log=False, **kwargs): # Add new arguments, if needed, e.g. base model, DEVICE
        super().__init__(data, self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.DEVICE = DEVICE
        self.log = log
    
    def criterion_fn(self, text: str):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.DEVICE)
            logits = self.model(**tokenized).logits[:, :-1]
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

            ranks = ranks.float() + 1  # convert to 1-indexed rank
            if self.log:
                ranks = torch.log(ranks)

            return ranks.float().mean().item()