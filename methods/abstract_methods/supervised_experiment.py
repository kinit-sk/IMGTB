from methods.abstract_methods.experiment import Experiment
from methods.utils import timeit, cal_metrics

import evaluate
import transformers
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from tqdm import tqdm
import numpy as np
import os
import time
import datetime
import pandas as pd
import gc

from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
import torch.nn.functional as F
import bitsandbytes as bnb

"""
Arguments:
    model - huggingfacehub identifier or local filepath to model weights
    cache_dir - path to directory to store cached models from huggingfacehub in
    batch_size - batch size for inference
    DEVICE - device to run inference/finetuning on ("cpu" or "cuda")
    finetune - boolean
    num_labels - size of output layer, number of output labels, 2 for binary classification
    label2id - similar to num_labels, specifies labels/ids for model on loading
    id2label - similar to num_labels, specifies labels/ids for model on loading
    epochs - number of epochs for finetuning
    model_output_machine_label - label that the model uses to indicate/label machine-generated text in output
    finetuning_batch_size - batch size for finetuning
    learning_rate - used in finetuning
    training_arguments - any additional named optional parameters for huggingface transformers library TrainingArguments class
    do_LoRA - boolean (modifies finetuning process to use LoRA PEFT technique)
    LoRA_params - any addional named optional parameters for huggingface peft library PeftConfig class
    bnb_quantization_config - any additional named optional parameters for huggingface transformers library BitsAndBytesConfig class
"""

F1_METRIC = evaluate.load("f1")

class CustomTrainer(transformers.Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits[:,1], labels.to(torch.float32))#, pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss

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
        self.finetuning_batch_size = config.get("finetuning_batch_size", 8)
        self.learning_rate = config.get("learning_rate", 2e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.training_arguments = config.get("training_arguments", {})
        self.do_lora = config.get("do_LoRA", False)
        self.lora_params = config.get("LoRA_params", {})
        self.bnb_quantization_config = config.get("bnb_quantization_config")
        self.label2id = config.get("label2id", {"human": 0, "machine": 1})
        self.id2label = config.get("id2label", {0:"human", 1:"machine"})
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
            quantization_config=transformers.BitsAndBytesConfig(self.bnb_quantization_config) if self.bnb_quantization_config is not None else None
        ).to(self.DEVICE)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model, 
            cache_dir=self.cache_dir
        )

        #DM added
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        detector.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
        try:
            detector.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
        except:
            print("Warning: Exception occured while setting pad_token_id")
        

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
                                  max_length=512)
    

def compute_metrics(eval_pred, metric_name="f1", average="micro"):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(F1_METRIC.compute(predictions=predictions, references = labels, average=average))

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

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    checkpoints_path = config["checkpoints_path"] + config["name"].replace("/", "-") + "-finetuned-" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    # create Trainer 
    training_args = transformers.TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=config.get("learning_rate", 2e-5),
        per_device_train_batch_size=config.get("finetuning_batch_size", 8),
        per_device_eval_batch_size=config.get("finetuning_batch_size", 8),
        num_train_epochs=config.get("epochs", 3),
        evaluation_strategy="no",
        save_strategy="no",
        load_best_model_at_end=True,
        **config.get("training_arguments", {})
    )

    if config.get("do_LoRA", False):
        lora_params = config.get("lora_params", {})
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            **lora_params
            )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        endpoints="wandb"
    )

    if config.get("do_LoRA", False):
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    trainer.model.save_pretrained(best_model_path, safe_serialization=True, max_shard_size="2GB")

    # Clear memory
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

