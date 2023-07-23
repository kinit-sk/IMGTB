class Experiment:
    def __init__(self, data, name):
        self.data = data
        self.name = name
    
    def run(self):
        raise NotImplementedError("Attempted to call an abstract method.")