from tqdm import tqdm
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
from methods.abstract_methods.experiment import Experiment
from methods.utils import get_rank, get_ll, get_entropy, cal_metrics, timeit, get_clf_results

class MetricBasedMashup(Experiment):
    
    def __init__(self, data, model, tokenizer, DEVICE, **kwargs):
        name = __class__.__name__
        super().__init__(data, name)
        self.model = model
        self.tokenizer = tokenizer
        self.DEVICE = DEVICE
    
    def criterion_fn(self, text: str):
        return np.array([get_rank(text, self.model, self.tokenizer, self.DEVICE),
                get_rank(text, self.model, self.tokenizer, self.DEVICE, log=True),
                get_ll(text, self.model, self.tokenizer, self.DEVICE),
                get_entropy(text, self.model, self.tokenizer, self.DEVICE)])

    @timeit
    def run(self):
        torch.manual_seed(0)
        np.random.seed(0)

        # get train data
        train_text = self.data['train']['text']
        train_label = self.data['train']['label']
        t1 = time.time()
        train_criterion = [self.criterion_fn(train_text[idx]) 
                           for idx in tqdm(range(len(train_text)), desc="Computing metrics on train partition")]
        x_train = np.array(train_criterion)
        y_train = np.expand_dims(train_label, axis=-1)

        test_text = self.data['test']['text']
        test_label = self.data['test']['label']
        test_criterion = [self.criterion_fn(test_text[idx])
                        for idx in tqdm(range(len(test_text)), desc="Computing test metrics on test partition")]
        x_test = np.array(test_criterion)
        y_test = np.expand_dims(test_label, axis=-1)
        
        model = build_FFNN()
        num_epochs = 30
        batch_size = 32
        train_history = model.fit(x_train, y_train.astype(np.float32), epochs=num_epochs, batch_size=batch_size, validation_split=0.2, shuffle=True)
        training_phase_data = process_training_phase_data(train_history, num_epochs=num_epochs)

        train_pred_prob = model.predict(x_train, batch_size=batch_size).reshape((-1,))
        test_pred_prob = model.predict(x_test, batch_size=batch_size).reshape((-1,))
        train_pred = np.array([1 if prob >= 0.5 else 0 for prob in train_pred_prob])
        test_pred = np.array([1 if prob >= 0.5 else 0 for prob in test_pred_prob])
        
        acc_train, precision_train, recall_train, f1_train, auc_train = cal_metrics(train_label, train_pred, train_pred_prob)
        acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(test_label, test_pred, test_pred_prob)

        print(f"{self.name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
        print(f"{self.name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

        return {
            'name': f'{self.name}',
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
            },
            'training_phase_data': training_phase_data
        }

def build_FFNN():
    model = Sequential()

    model.add(Dense(128, kernel_initializer='normal', input_dim=4, activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
   
    model.compile(loss=BinaryCrossentropy(), optimizer='adam', metrics=[F1Score(), Accuracy(), Precision(), Recall(), AUC()])
    model.summary()
    return model
    

def process_training_phase_data(history, num_epochs):
    data = [{'epoch': str(idx),
        'loss': str(history.history['loss'][idx]),
        'f1_score': str(history.history['f1_score'][idx][0]),
        'accuracy': str(history.history['accuracy'][idx]),
        'precision': str(history.history['precision'][idx]),
        'recall': str(history.history['recall'][idx]),
        'auc': str(history.history['auc'][idx]),
        'val_loss': str(history.history['val_loss'][idx]),
        'val_f1_score': str(history.history['val_f1_score'][idx][0]),
        'val_accuracy': str(history.history['val_accuracy'][idx]),
        'val_precision': str(history.history['val_precision'][idx]),
        'val_recall': str(history.history['val_recall'][idx]),
        'val_auc': str(history.history['val_auc'][idx])
    } 
        for idx in range(num_epochs)]
    return data