import numpy as np
import torch
import torch.nn.functional as F
import time
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import F1Score, Accuracy, Precision, Recall, AUC
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from methods.utils import timeit, get_clf_results
from methods.abstract_methods.experiment import Experiment
from methods.utils import get_rank, get_ll, get_entropy

class MetricBasedMashup(Experiment):
    
    def __init__(self, data, model, tokenizer, DEVICE, **kwargs):
        name = __class__.__name__
        super().__init__(data, name)
        self.model = model
        self.tokenizer = tokenizer
        self.DEVICE = DEVICE
    
    def criterion_fn(self, text: str):
        return (get_rank(text, self.model, self.tokenizer, self.DEVICE),
                get_rank(text, self.model, self.tokenizer, self.DEVICE, log=True),
                get_ll(text, self.model, self.tokenizer, self.DEVICE),
                get_entropy(text, self.model, self.tokenizer, self.DEVICE))

    @timeit
    def run(self):
        torch.manual_seed(0)
        np.random.seed(0)

        # get train data
        train_text = self.data['train']['text']
        train_label = self.data['train']['label']
        t1 = time.time()
        train_criterion = [self.criterion_fn(train_text[idx])
                        for idx in range(len(train_text))]
        x_train = np.array(train_criterion)
        #x_train = np.expand_dims(x_train, axis=-1)
        y_train = np.expand_dims(train_label, axis=-1)

        test_text = self.data['test']['text']
        test_label = self.data['test']['label']
        test_criterion = [self.criterion_fn(test_text[idx])
                        for idx in range(len(test_text))]
        x_test = np.array(test_criterion)
        #x_test = np.expand_dims(x_test, axis=-1)
        y_test = np.expand_dims(test_label, axis=-1)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        
        # Train and test FFNN 
        #train_pred, test_pred, train_pred_prob, test_pred_prob, train_res, test_res = get_clf_results(x_train, y_train, x_test, y_test)
        model = build_FFNN()
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)
        train_res = model.fit(np.array(x_train), np.array(y_train).astype(np.float32), epochs=50, batch_size=25, validation_data=(np.array(x_val), np.array(y_val).astype(np.float32)), shuffle=True)
        test_res = model.evaluate(np.array(x_test), np.array(y_test).astype(np.float32), batch_size=25)
        print(train_res)
        print(test_res)
        exit(0)
        acc_train, precision_train, recall_train, f1_train, auc_train = train_res["Accuracy"], train_res["Precision"], train_res["Recall"], train_res["F1"]
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res

        print(f"{self.name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
        print(f"{self.name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

        return {
            'name': f'{self.name}_threshold',
            'input_data': self.data,
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

def build_FFNN():
    model = Sequential()

    model.add(Dense(16, kernel_initializer='normal',input_dim = 4, activation='tanh'))
    model.add(Dense(8, kernel_initializer='normal',activation='relu'))
    model.add(Dense(2, kernel_initializer='normal',activation='relu'))
   
    model.compile(loss=BinaryCrossentropy(), optimizer='adam', metrics=[F1Score(), Accuracy(), Precision(), Recall(), AUC()])
    model.summary()
    return model
    

def visualize():
    import hiplot as hip

    num_epochs = 50
    data = [{'epoch': idx,
        'loss': history.history['loss'][idx],
        'RMSE': history.history['root_mean_squared_error'][idx],
        'MAE': history.history['mean_absolute_error'][idx],
        'val_loss': history.history['val_loss'][idx],
        'val_RMSE': history.history['val_root_mean_squared_error'][idx],
        'val_MAE': history.history['val_mean_absolute_error'][idx],
    } 
        for idx in range(num_epochs)]

    hip.Experiment.from_iterable(data).display()