from methods.abstract_methods.experiment import Experiment

class GenericPlainTemplate(Experiment):
     def __init__(self, data, **kwargs): # Add extra parameters if needed
        name = self.__class__.__name__ # Set your own name or leave it set to the class name
        super().__init__(data, name)
        
     def run(self):
        """
        Run the experiment on the provided data, evaluate and return results similar to:
        return {
            'name': f'{self.name}',
            'predictions': {'train': train_criterion, 'test': test_criterion},
            'general': {
                'acc_train': acc_train,
                'precision_train': precision_train,
                'recall_train': recall_train,
                'f1_train': f1_train,
                'auc_train': auc_train,
                'acc_test': acc_test,
                'precision_test': precision_test,
                'recall_test': recall_test,
                'f1_test': f1_test,
                'auc_test': auc_test,
            }
        }
        """
        pass

