from methods.abstract_methods.experiment import Experiment
from methods.utils import cal_metrics, timeit, move_model_to_device

import transformers
import evaluate
import datasets
import torch
import numpy as np
import pandas as pd
from ftlangdetect import detect
from tqdm import tqdm
import datetime

from transformers import EarlyStoppingCallback
import sklearn.model_selection
from sklearn.metrics import auc, roc_curve

import time
import gc
import os

"""
Arguments:
    DEVICE - string representing device to run computations on ("cpu" or "cuda")
    cache_dir - path to a directory to store cache
    checkpoints_path - path to a directory to store checkpoints of finetuned models
    finetune - boolean
    model_output_machine_label - string representing the machine label (of model output)
    per_language_models - dictionary of (string representing language: path to model)
    base_model_name - model to be used for finetuning of language, if not specified in per_language_models differently
    language_column - string representing the name of the column in data that labels language of example
    early_stopping - boolean
    batch_size - batch size for inference
    finetuning_batch_size - batch size for finetuning
    threshold_calibration - boolean
    thresholds - Dict[str, float] - mapping languages to thresholds

"""

DEFAULT_THRESHOLD = 0.5

class PerLanguageExpertsEnsemble(Experiment):

     def __init__(self, data, config):
        name = self.__class__.__name__
        super().__init__(data, name)
        self.base_model_name = config["base_model_name"]
        self.cache_dir = config["cache_dir"]
        self.DEVICE = config["DEVICE"]
        self.config = config
        self.batch_size = config["batch_size"]
        self.finetuning_batch_size = config.get("finetuning_batch_size", 16)
        self.model_output_machine_label = config["model_output_machine_label"]
        self.per_language_models = config.get("per_language_models", {})
        self.do_finetune = config["finetune"]
        self.checkpoints_path = config["checkpoints_path"]
        self.language_column = config.get("language_column", "language")
        self.early_stopping = config.get("early_stopping", False)
        self.threshold_calibration = config.get("threshold_calibration", False)
        self.thresholds = config.get("thresholds", {})

     
     def get_predictions_for_multiple(self, data):
         if self.language_column not in data.keys():
             data[self.language_column] = [lang if (lang:=get_language(text)) in self.per_language_models.keys() else "unknown" for text in tqdm(data["text"], desc="Running language identification on input data")]
         languages = data[self.language_column]
         weights = get_weights(languages, self.per_language_models)
         results = []
         for clf in self.load_models():
             pos_bit = set_pos_bit(clf["model"], self.model_output_machine_label)
             results.append(self.get_predictions_for_single(clf["name"], clf["model"], clf["tokenizer"], data["text"], pos_bit))
             free_model_memory(clf)
         weighted_averages = np.multiply(np.array(weights).transpose(), np.array(results))
         preds_for_each_lang = {lang:preds for lang, preds in zip(self.per_language_models.keys(), results)} 
         return np.sum(weighted_averages, axis=0), preds_for_each_lang

     def get_predictions_for_single(self, name, model, tokenizer, data, pos_bit):
         with torch.no_grad():
                preds = []
                for start in tqdm(range(0, len(data),self.batch_size), desc=f"Evaluating data with language-specific model: {name}"):
                    end = min(start + self.batch_size, len(data))
                    batch_data = data[start:end]
                    batch_data = tokenizer(
                        batch_data,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.DEVICE)
                    preds.extend(model(**batch_data).logits.softmax(-1)[:, pos_bit].tolist())

         return preds
     
     def load_models(self):
         for name, filepath in self.per_language_models.items():
             model = transformers.AutoModelForSequenceClassification.from_pretrained(filepath, cache_dir=self.cache_dir)
             tokenizer = transformers.AutoTokenizer.from_pretrained(filepath, cache_dir=self.cache_dir)
             move_model_to_device(model, self.DEVICE)
             yield {"name": name, "model": model, "tokenizer": tokenizer}

    
     def finetune(self):
         ID2LABEL = {0: "human", 1: "machine"}
         LABEL2ID = {"human": 0, "machine": 1}
         language_splits = split_by_language(pd.DataFrame(self.data["train"]), self.language_column, self.per_language_models)
         language_splits.update({"unknown": self.data["train"]}) # Whole dataset for training of the multilingual detector for unknown languages
         for lang, df in language_splits.items():
    
             model_name = self.per_language_models.get(lang, self.base_model_name)
             
             spec_checkpoints_path = os.path.join(self.checkpoints_path, model_name.replace("/", "-") + "-" + lang + "-finetuned-" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
             config = {
                 "name": model_name,
                 "checkpoints_path": spec_checkpoints_path,
                 "language": lang,
                 "global_config": self.config
             }

             print("Loading model", model_name)
             model = transformers.AutoModelForSequenceClassification.from_pretrained(
                     model_name,
                     num_labels=len(LABEL2ID),
                     label2id=LABEL2ID,
                     id2label=ID2LABEL,
                     cache_dir=self.cache_dir
             ).to(self.DEVICE)
             tokenizer = transformers.AutoTokenizer.from_pretrained(
                         model_name, 
                         cache_dir=self.cache_dir
             )

             print("Finetuning", model_name, "on language", lang)
             finetune_model(df, model, tokenizer, config)
             print("Finished finetuning", model_name, "on language", lang)
             print("Storing model at path", spec_checkpoints_path)
             self.per_language_models[lang] = spec_checkpoints_path + "/best"
    
         # Clean up
         del model
         del tokenizer
         gc.collect()
         torch.cuda.empty_cache()
    
     def calibrate_thresholds(self, metrics_per_lang):
         df_metrics = pd.DataFrame(metrics_per_lang)
         df_metrics = df_metrics.add_prefix("language-pred-prob-")
         data = pd.concat([pd.DataFrame(self.data["train"]), df_metrics], axis="columns")
         language_splits = split_by_language(pd.DataFrame(data), self.language_column, self.per_language_models)
         language_splits["unknown"] = data # Whole dataset for training of the multilingual detector for unknown languages
         optimal_thresholds = {}
         for lang, df in language_splits.items():
             if lang not in self.per_language_models.keys():
                 continue
             fpr, tpr, thresholds = roc_curve(df["label"], df["language-pred-prob-"+lang])
             th_optim_idx = np.argmax(tpr - fpr)
             th_optim2_idx = np.argmin(np.abs(fpr+tpr-1))
             if fpr[th_optim_idx] < fpr[th_optim2_idx]:
                 optimal_thresholds[lang] = thresholds[th_optim_idx]
             else:
                 optimal_thresholds[lang] = thresholds[th_optim2_idx]
         self.thresholds = optimal_thresholds
    
     @timeit
     def run(self):
        start_time = time.time()
        
        if not self.do_finetune and "unknown" not in self.per_language_models.keys():
            raise ValueError("No language model specified for language 'unknown'. When not finetuning and running only inference, please specify all per-language models for each language explicitly in config.")

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print(f"Using cache dir {self.cache_dir}")
        
        if "cuda" in self.DEVICE and not torch.cuda.is_available():
            print(f'Setting default device to cpu. Cuda is not available.')
            self.DEVICE = "cpu"
        
        if self.do_finetune:
            self.finetune()
        
        torch.manual_seed(0)
        np.random.seed(0)

        train_data = self.data['train']
        train_label = self.data['train']['label']
        y_train_pred_prob = np.array([])
        y_train = []
        y_train_pred = []
        machine_prob_by_lang_train = []
        acc_train, precision_train, recall_train, f1_train, auc_train, specificity_train = -1, -1, -1, -1, -1, -1
        if len(train_data["text"]) != 0:
            y_train_pred_prob, machine_prob_by_lang_train = self.get_predictions_for_multiple(train_data)
            y_train = train_label
            y_train_pred = [round(_) for _ in y_train_pred_prob]
            train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
            acc_train, precision_train, recall_train, f1_train, auc_train, specificity_train = train_res

        if self.threshold_calibration:
            print("Running threshold calibration")
            self.calibrate_thresholds(machine_prob_by_lang_train)
            print("Thresholds:", self.thresholds)

        test_data = self.data['test']
        test_label = self.data['test']['label']
        y_test_pred_prob, machine_prob_by_lang_test = self.get_predictions_for_multiple(test_data)
        y_test = test_label

        y_test_pred = [0 if prob < get_threshold(self.thresholds, self.data["test"], self.language_column, idx) else 1 for idx, prob in enumerate(y_test_pred_prob)]
        y_test = test_label

        test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
        acc_test, precision_test, recall_test, f1_test, auc_test, specificity_test = test_res
        print(f"{self.name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}, specificity_train: {specificity_train}")
        print(f"{self.name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}, specificity_test: {specificity_test}")
 

        
        return {
            'name': 'PerLanguageExpertsEnsemble',
            'type': 'ensemble',
            'input_data': {"train": self.data["train"], "test": self.data["test"]},
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'machine_prob': {'train': y_train_pred_prob.tolist(), 'test': y_test_pred_prob.tolist()},
            'machine_prob_by_lang': {'train': machine_prob_by_lang_train, 'test': machine_prob_by_lang_test},
            'running_time_seconds': time.time() - start_time,
            'metrics_results': {
                'train': {
                    'acc': acc_train,
                    'precision': precision_train,
                    'recall': recall_train,
                    'f1': f1_train,
                    'specificity': specificity_train
                },
                'test': {
                    'acc': acc_test,
                    'precision': precision_test,
                    'recall': recall_test,
                    'f1': f1_test,
                    'specificity': specificity_test
                }
            },
            "config": self.config
        }
    
def get_threshold(thresholds, data, language_column, idx):
    return thresholds.get(data[language_column][idx], DEFAULT_THRESHOLD)


def free_model_memory(clf):
    del clf["model"]
    del clf["tokenizer"]
    torch.cuda.empty_cache()
    gc.collect()



def set_pos_bit(model,model_output_machine_label: str, pos_bit=0) -> int:
    """Try to find the right model output label (it's index) to evaluate

    Args:
        model (transformers model): supervised model loaded through hugging face
        model_output_machine_label: label set by user (or default) 
        pos_bit: default output label index

    Returns:
        pos_bit: output label index to be set
    """
    if len(model.config.id2label.keys()) == 1:
        return 0
    elif "machine" in model.config.label2id.keys():
        return model.config.label2id["machine"]
    elif "fake" in model.config.label2id.keys():
        return model.config.label2id["fake"]
    elif isinstance(model_output_machine_label, str):
        return model.config.label2id[model_output_machine_label]
    elif isinstance(model_output_machine_label, int):
        return model_output_machine_label
    else:
        return pos_bit


def get_language(text):
    text = text.lower()
    res = detect(text=text.replace('\n', ' '), low_memory=False)
    if res['score'] > 0.5 and res["lang"]: return res['lang']
    return 'unknown'


def get_weights(languages, per_language_models):
    return [[1 if model_lang == lang else 0 for model_lang in per_language_models.keys()] for lang in languages]


def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], 
                                  padding=True,
                                  truncation=True,
                                  max_length=512,)


def compute_metrics(eval_pred, metric_name="f1", average="micro"):

    metric = evaluate.load(metric_name)

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(metric.compute(predictions=predictions, references = labels, average=average))

    return results


def finetune_model(data, model, tokenizer, config):
    data = pd.DataFrame(data)
    data["label"] = data["label"].astype(int)
    data_train, data_val = sklearn.model_selection.train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)

    # pandas dataframe to huggingface Dataset
    train_dataset = datasets.Dataset.from_dict(data_train)
    valid_dataset = datasets.Dataset.from_dict(data_val)

    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    callbacks = []
    if config["global_config"].get("early_stopping", False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["global_config"].get("early_stopping_patience", 5)))
    
    # create Trainer 
    training_args = transformers.TrainingArguments(
        output_dir=config["checkpoints_path"],
        learning_rate=2e-5,
        per_device_train_batch_size=config["global_config"].get("finetuning_batch_size", 16),
        per_device_eval_batch_size=config["global_config"].get("finetuning_batch_size", 16),
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        gradient_accumulation_steps=4,
        eval_steps=80,
        save_steps=800,
        logging_steps=160,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()
    print("Results:", trainer.evaluate())

    # Save best model
    best_model_path = config["checkpoints_path"] +'/best/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    trainer.save_model(best_model_path)
    
    # Save training history
    pd.DataFrame(trainer.state.log_history).to_csv(config["checkpoints_path"] + "/history.csv")
    
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
def split_by_language(df: pd.DataFrame, language_column, per_language_models):
    if not language_column in df.columns:
        df[language_column] = [lang if (lang:=get_language(text)) in per_language_models.keys() else "unknown" for text in tqdm(df["text"])]
    print("Detected following languages: ", df[language_column].unique())
    # Split into subsets by 
    return {lang: subdf for lang, subdf in df.groupby(df[language_column])}
