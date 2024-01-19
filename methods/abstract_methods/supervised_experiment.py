from methods.abstract_methods.experiment import Experiment
from methods.utils import timeit, cal_metrics
import evaluate
import transformers
from transformers import AdamW, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
import datetime
import pandas as pd
import gc


class SupervisedExperiment(Experiment):
    def __init__(
        self,
        data,
        name,
        model: str,
        config,
    ):
        super().__init__(data, name)
        self.model = model
        self.cache_dir = config["cache_dir"]
        self.batch_size = config["batch_size"]
        self.DEVICE = config["DEVICE"]
        self.pos_bit = 0
        self.finetune = config["finetune"]
        self.num_labels = config["num_labels"]
        self.epochs = config["epochs"]
        self.model_output_machine_label = config["model_output_machine_label"]
        self.config = config

    @timeit
    def run(self):
        start_time = time.time()
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Using cache dir {self.cache_dir}")
        if "cuda" in self.DEVICE and not torch.cuda.is_available():
            print(f"Setting default device to cpu. Cuda is not available.")
            self.DEVICE = "cpu"

        print(f"Beginning supervised evaluation with {self.model}...")
        detector = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model,
            num_labels=self.num_labels,
            cache_dir=self.cache_dir,
            ignore_mismatched_sizes=True,
        ).to(self.DEVICE)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model, cache_dir=self.cache_dir
        )

        if self.finetune:
            fine_tune_model(
                self.data["train"],
                detector,
                tokenizer,
                self.config
            )

        train_text = self.data["train"]["text"]
        train_label = self.data["train"]["label"]
        test_text = self.data["test"]["text"]
        test_label = self.data["test"]["label"]

        # detector.save_pretrained(".cache/lm-d-xxx", from_pt=True)
        
        self.pos_bit = set_pos_bit(detector, self.model_output_machine_label, self.pos_bit)
        
        if self.num_labels == 2:
            train_preds = get_supervised_model_prediction(
                detector,
                tokenizer,
                train_text,
                self.batch_size,
                self.DEVICE,
                self.pos_bit,
            )
            test_preds = get_supervised_model_prediction(
                detector,
                tokenizer,
                test_text,
                self.batch_size,
                self.DEVICE,
                self.pos_bit,
            )
        else:
            train_preds = get_supervised_model_prediction_multi_classes(
                detector,
                tokenizer,
                train_text,
                self.batch_size,
                self.DEVICE,
                self.pos_bit,
            )
            test_preds = get_supervised_model_prediction_multi_classes(
                detector,
                tokenizer,
                test_text,
                self.batch_size,
                self.DEVICE,
                self.pos_bit,
            )

        y_train_pred_prob = train_preds
        y_train_pred = [round(_) for _ in y_train_pred_prob]
        y_train = train_label

        y_test_pred_prob = test_preds
        y_test_pred = [round(_) for _ in y_test_pred_prob]
        y_test = test_label

        train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
        test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
        acc_train, precision_train, recall_train, f1_train, auc_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res
        print(
            f"{self.model} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}"
        )
        print(
            f"{self.model} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}"
        )

        # Clean up
        del detector
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "name": self.model,
            'type': 'supervised',
            "input_data": self.data,
            "predictions": {"train": y_train_pred, "test": y_test_pred},
            "machine_prob": {"train": y_train_pred_prob, "test": y_test_pred_prob},
            'running_time_seconds': time.time() - start_time,
            "metrics_results": {
                "train": {
                    "acc": acc_train,
                    "precision": precision_train,
                    "recall": recall_train,
                    "f1": f1_train,
                },
                "test": {
                    "acc": acc_test,
                    "precision": precision_test,
                    "recall": recall_test,
                    "f1": f1_test,
                }
            },
            "config": self.config
        }


def set_pos_bit(model,model_output_machine_label: str, pos_bit: int) -> int:
    """Try to find the right model output label (it's index) to evaluate

    Args:
        model (transformers model): supervised model loaded through hugging face
        model_output_machine_label: label set by user (or default) 
        pos_bit: default output label index

    Returns:
        pos_bit: output label index to be set
    """
    if len(model.config.id2label.keys()) == 1:
        print("0")
        return 0
    elif "machine" in model.config.label2id.keys():
        print("1")
        return model.config.label2id["machine"]
    elif "fake" in model.config.label2id.keys():
        print("2")
        return model.config.label2id["fake"]
    elif isinstance(model_output_machine_label, str):
        print("3")
        return model.config.label2id[model_output_machine_label]
    elif isinstance(model_output_machine_label, int):
        print("4")
        return model_output_machine_label
    else:
        print("5")
        return pos_bit


def get_supervised_model_prediction(
    model, tokenizer, data, batch_size, DEVICE, pos_bit=0
):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating real"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            preds.extend(model(**batch_data).logits.softmax(-1)[:, pos_bit].tolist())
    return preds


def get_supervised_model_prediction_multi_classes(
    model, tokenizer, data, batch_size, DEVICE, pos_bit=0
):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating real"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            preds.extend(torch.argmax(model(**batch_data).logits, dim=1).tolist())
    return preds


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


def fine_tune_model(data, model, tokenizer, config):
    data = pd.DataFrame(data)
    data["label"] = data["label"].astype(int)
    data_train, data_val = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_dict(data_train)
    valid_dataset = Dataset.from_dict(data_val)

    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    checkpoints_path = config["checkpoints_path"] + config["name"].replace("/", "-") + "-finetuned-" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        evaluation_strategy="no",
        save_strategy="no",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)
    
    # Clear memory
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

