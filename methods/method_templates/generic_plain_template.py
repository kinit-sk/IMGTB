from methods.abstract_methods.experiment import Experiment

class GenericPlainTemplate(Experiment):
     def __init__(self, data, config):
        name = self.__class__.__name__
        super().__init__(data, name)
        
     def run(self):
        """
        Run the experiment on the provided data, evaluate and return results similar to:
        return {
            'name': name,
            'input_data': data, 
            'predictions': {'train': train_pred, 'test': test_pred},
            'machine_prob': {'train': train_pred_prob, 'test': test_pred_prob},
            'metrics_results': {
                'train': {
                    'acc': acc_train,
                    'precision': precision_train,
                    'recall': recall_train,
                    'f1': f1_train
                },
                'test': {
                    'acc': acc_test,
                    'precision': precision_test,
                    'recall': recall_test,
                    'f1': f1_test
                }
            }
        }
        """
        pass

