from methods.abstract_methods.experiment import Experiment
from methods.utils import timeit, cal_metrics
import transformers
from transformers import AdamW
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class SupervisedExperiment(Experiment):
    def __init__(
        self,
        data,
        name,
        model: str,
        config,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=3,
    ):
        super().__init__(data, name)
        self.model = model
        self.cache_dir = config["cache_dir"]
        self.batch_size = config["batch_size"]
        self.DEVICE = config["DEVICE"]
        self.pos_bit = pos_bit
        self.finetune = finetune
        self.num_labels = num_labels
        self.epochs = epochs
        self.model_output_machine_label = config["model_output_machine_label"]

    @timeit
    def run(self):

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
                detector,
                tokenizer,
                self.data,
                self.batch_size,
                self.DEVICE,
                self.pos_bit,
                self.num_labels,
                epochs=self.epochs,
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

        # free GPU memory
        del detector
        torch.cuda.empty_cache()

        return {
            "name": self.model,
            "input_data": self.data,
            "predictions": {"train": y_train_pred, "test": y_test_pred},
            "machine_prob": {"train": y_train_pred_prob, "test": y_test_pred_prob},
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
                },
            },
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
            print(model.config.id2label)
            print(model(**batch_data))
            print(model(**batch_data).logits)
            print(model(**batch_data).logits.softmax(-1)[:, pos_bit])
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


def fine_tune_model(
    model, tokenizer, data, batch_size, DEVICE, pos_bit=1, num_labels=2, epochs=3
):

    # https://huggingface.co/transformers/v3.2.0/custom_datasets.html

    train_text = data["train"]["text"]
    train_label = data["train"]["label"]
    test_text = data["test"]["text"]
    test_label = data["test"]["label"]

    if pos_bit == 0 and num_labels == 2:
        train_label = [1 if _ == 0 else 0 for _ in train_label]
        test_label = [1 if _ == 0 else 0 for _ in test_label]

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_label)
    test_dataset = CustomDataset(test_encodings, test_label)

    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
    model.eval()
