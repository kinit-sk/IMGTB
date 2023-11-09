import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import classification_report
import seaborn as sns
from math import ceil, sqrt
import os
import traceback
import sys
import json
from collections import Counter
from typing import List

########################################
#            Helper functions          #
########################################

def _find_interval_label(labeled_intervals, num):
  for label, interval in labeled_intervals.items():
    if num in interval:
      return label


########################################
#           Analysis methods           #
########################################

def analyze_test_metrics(results_list, save_path, is_interactive: bool):
  for dataset_name, dataset_results in results_list.items():
    results = pd.Series()
    for detector in dataset_results:
      results = pd.concat([results, pd.DataFrame({'Detector': detector["name"], 
                                                  'Accuracy': detector["metrics_results"]["test"]['acc'], 
                                                  'Precision': detector["metrics_results"]["test"]['precision'], 
                                                  'Recall': detector["metrics_results"]["test"]["recall"], 
                                                  'F1-score': detector["metrics_results"]["test"]["f1"]}, index=[0])], copy=False, ignore_index=True)  
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{dataset_name} dataset", fontsize=16)
    
    rows = 2
    cols = 2
    
    fig.add_subplot(rows, cols, 1)
    ax = sns.barplot(results, x="Detector", y="F1-score")
    ax.set(ylim=(0,1))
    plt.xticks(rotation=25, ha="right")
    
    fig.add_subplot(rows, cols, 2)
    ax = sns.barplot(results, x="Detector", y="Accuracy")
    ax.set(ylim=(0,1))
    plt.xticks(rotation=25, ha="right")

    fig.add_subplot(rows, cols, 3)
    ax = sns.barplot(results, x="Detector", y="Precision")
    ax.set(ylim=(0,1))
    plt.xticks(rotation=25, ha="right")
    
    fig.add_subplot(rows, cols, 4)
    ax = sns.barplot(results, x="Detector", y="Recall")
    ax.set(ylim=(0,1))
    plt.xticks(rotation=25, ha="right")

    fig.tight_layout()  
    plt.savefig(os.path.join(save_path, f"{dataset_name}_metrics_analysis.png"))
    if is_interactive:
      plt.show()

def _get_num_of_tokens(data: List[str]):
  word_count = 0
  for text_data in data:
    spaces = text_data.count(' ')
    tabs = text_data.count('\t')
    newlines = text_data.count('\n')
    word_count += spaces+tabs+newlines
  return word_count


def analyze_running_time(results_list, save_path: str, is_interactive: bool) -> None:
  """
  This function requires each experiment in the results list 
  to have a 'running_time_seconds' item in its benchmark results.
  """
  if len(results_list.keys()) > 1:
    analyze_running_time_over_multiple_datasets(results_list, save_path, is_interactive)
  
  for dataset_name, dataset_results in results_list.items():
    results = pd.Series()
    data_word_count = _get_num_of_tokens(dataset_results[0]["input_data"]["train"]["text"] + 
                                         dataset_results[0]["input_data"]["test"]["text"])
    for detector in dataset_results:
      results = pd.concat([results, pd.DataFrame({'Detector': detector["name"], 
                                                  'RunningTimeSeconds': detector["running_time_seconds"],
                                                  'RunningTimeSecondsPerWord': detector["running_time_seconds"] / data_word_count}, index=[0])], copy=False, ignore_index=True)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{dataset_name} dataset", fontsize=16)
    
    rows = 1
    cols = 2
    
    fig.add_subplot(rows, cols, 1)
    sns.barplot(results, x="Detector", y="RunningTimeSeconds")
    plt.xticks(rotation=25, ha="right")
    
    fig.add_subplot(rows, cols, 2)
    sns.barplot(results, x="Detector", y="RunningTimeSecondsPerWord")
    plt.xticks(rotation=25, ha="right")
    
    plt.savefig(os.path.join(save_path, f"{dataset_name}_running_time_comparison_analysis.png"))
    if is_interactive:
      plt.show()
  

def analyze_running_time_over_multiple_datasets(results_list, save_path: str, is_interactive: bool) -> None:
  """
  This function requires each experiment in the results list 
  to have a 'running_time_seconds' item in its benchmark results.
  """
  results = pd.DataFrame()
  for dataset_name, dataset_results in results_list.items():
      data_word_count = _get_num_of_tokens(dataset_results[0]["input_data"]["train"]["text"] + 
                                           dataset_results[0]["input_data"]["test"]["text"])
      for detector in dataset_results:
        results = pd.concat([results, pd.DataFrame({'Detector': detector["name"],
                                                    'Dataset': dataset_name,
                                                    'RunningTimeSeconds': detector["running_time_seconds"],
                                                    'RunningTimeSecondsPerWord': detector["running_time_seconds"] / data_word_count}, index=[0])], copy=False, ignore_index=True)
  # Raw running time seconds (not normalized)
  fig = plt.figure(figsize=(10, 10))
  fig.suptitle("Running time (seconds) over multiple datasets", fontsize=16)
  
  rows, cols = 1, 2

  fig.add_subplot(rows, cols, 1)
  ax = sns.lineplot(results, x="Dataset", y="RunningTimeSeconds", hue="Detector")
  ax.set(title="Running time (sec) over multiple datasets")
  
  fig.add_subplot(rows, cols, 2)
  ax = sns.lineplot(results, x="Dataset", y="RunningTimeSecondsPerWord", hue="Detector")
  ax.set(title="Normalized (per word) running time (sec) over multiple datasets")
  
  
  fig.tight_layout()  
  plt.savefig(os.path.join(save_path, "running_time_over_multiple_datasets.png"), dpi=600)
  if is_interactive:
    plt.show()
      

def analyze_text_lengths(results_list, save_path, is_interactive: bool):
  for dataset_name, dataset_results in results_list.items():
    lengths_groups = [[0, 50], [50, 100], [100, 200], [200, 500], [500, 100000]]
    results = pd.DataFrame()
    for detector in dataset_results:
      detector_data = pd.DataFrame({"text": detector["input_data"]["test"]["text"], 
                                  "label": detector["input_data"]["test"]["label"], 
                                  "prediction": detector["predictions"]["test"],
                                  "machine_prob": detector["machine_prob"]["test"]})
      for length_group in lengths_groups:
        temp = detector_data[(detector_data['text'].str.len() >= length_group[0]) & (detector_data['text'].str.len() < length_group[1])]
        if len(temp) == 0: continue
        cr = classification_report(temp['label'], temp['prediction'], labels=[0, 1], target_names=['machine', 'human'], digits=4, output_dict=True, zero_division=0)
        results = pd.concat([results, pd.DataFrame({'Length group': str(length_group), 'Detector': detector["name"], 'F1-score': cr['weighted avg']['f1-score'], 'Human samples': cr['human']['support'], 'Machine samples': cr['machine']['support']}, index=[0])], copy=False, ignore_index=True) 

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(f"{dataset_name} dataset", fontsize=16)
    
    rows = 1
    cols = 2
    
    fig.add_subplot(rows, cols, 1)
    ax = sns.barplot(results, x="Length group", y="F1-score", hue="Detector")
    ax.set(ylim=(0, 1), xlabel="Length group (word count)")
    
    fig.add_subplot(rows, cols, 2)
    ax = sns.lineplot(results, x="Length group", y="F1-score", hue="Detector")
    ax.set(ylim=(0, 1), xlabel="Length group (word count)")
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, f'{dataset_name}_text_lengths_analysis.png'), dpi=600)
    if is_interactive:
      plt.show()

def analyze_pred_prob_hist(results_list, save_path, is_interactive: bool):
  """
  For each method evaluated on a given dataset, 
  show a histogram of prediction probability values 
  (probability that a text is machine-generated)
  """
  for dataset_name, dataset_results in results_list.items():
    results = pd.DataFrame()
    for detector in dataset_results:
      results = pd.concat([results, pd.DataFrame({detector["name"]: detector["machine_prob"]["test"]})], axis="columns", copy=False)  
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{dataset_name}/Prediction Probability Histograms", fontsize=16)
    
    rows = cols = ceil(sqrt(len(results.columns)))
    
    for i in range(len(results.columns)):
      column_name = results.columns[i]
      fig.add_subplot(rows, cols, i+1)
      ax = sns.histplot(results[column_name])
      ax.set(xlim=(0,1))
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, f"{dataset_name}_pred_prob_hist_analysis.png"), dpi=600)
    if is_interactive:
      plt.show()

def analyze_pred_prob_error_hist(results_list, save_path, is_interactive: bool):
  """
  For each method evaluated on a given dataset, 
  show a histogram of errors in prediction 
  (how far was the prediction from true label, how often)
  """
  for dataset_name, dataset_results in results_list.items():
    results = pd.DataFrame()
    for detector in dataset_results:
      results = pd.concat([results, pd.DataFrame(
        {detector["name"]: abs(np.array(detector["machine_prob"]["test"]) - np.array(detector["input_data"]["test"]["label"]))
         })], axis="columns", copy=False)  
    
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{dataset_name}/Prediction Probability Error Histograms", fontsize=16)
    
    rows = cols = ceil(sqrt(len(results.columns)))
    
    for i in range(len(results.columns)):
      column_name = results.columns[i]
      fig.add_subplot(rows, cols, i+1)
      ax = sns.histplot(results[column_name])
      ax.set(title=column_name, xlim=(0,1))
    
    fig.tight_layout()  
    plt.savefig(os.path.join(save_path, f"{dataset_name}_pred_prob_error_hist_analysis.png"), dpi=600)
    if is_interactive:
      plt.show()

def analyze_false_positives(results_list, save_path, is_interactive: bool):
  """
  TN - true negative
  PTN - partially true negative
  UNC - unclear
  PFP - partially false positive
  FP - false positive
  """
  PRED_PROB_INTERVALS = {
    "TN": pd.Interval(0, 0.2),
    "PTN": pd.Interval(0.2, 0.4),
    "UNC": pd.Interval(0.4, 0.6),
    "PFP": pd.Interval(0.6, 0.8),
    "FP": pd.Interval(0.8, 1)
  }
  
  for dataset_name, dataset_results in results_list.items():
    results = pd.DataFrame(columns=['Detector', 'TN', 'PTN', 'UNC', 'PFP', 'FP'])
    results = results.astype({"Detector": "object", 
                    "TN": "float64", 
                    "PTN": "float64", 
                    "UNC": "float64", 
                    "PFP": "float64", 
                    "FP": "float64"})
    for detector in dataset_results:
      pred_data = pd.DataFrame({
        "pred_prob": detector["machine_prob"]["test"], 
        "true_label": detector["input_data"]["test"]["label"]})
      pred_data_positive = pred_data[pred_data["true_label"] == 0]
      pred_data_positive["confusion_label"] = pred_data_positive.apply(lambda row: _find_interval_label(PRED_PROB_INTERVALS, row["pred_prob"]), axis="columns")
      
      counts = dict(Counter(list(pred_data_positive["confusion_label"])))
      counts_sum = sum(counts.values())
      if counts_sum == 0:
        print("Cannot perform false positives analysis on data with no negative samples. Exiting...")
        return
      
      results = pd.concat([results, pd.DataFrame(
        {"Detector": detector["name"], 
         "TN": counts.get("TN", 0) / counts_sum * 100,
         "PTN": counts.get("PTN", 0) / counts_sum * 100,
         "UNC": counts.get("UNC", 0) / counts_sum * 100,
         "PFP": counts.get("PFP", 0) / counts_sum * 100,
         "FP": counts.get("FP", 0) / counts_sum * 100}, index=[0])], axis="index", ignore_index=True)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    results.plot(stacked=True, ax=ax, kind="barh", x="Detector")
    ax.set(title=f"{dataset_name}/False Positives Analysis", xlim=(0,100))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    plt.savefig(os.path.join(save_path, f"{dataset_name}_false_positives_analysis.png"), dpi=600)
    if is_interactive:
      plt.show()  
  
  

def analyze_false_negatives(results_list, save_path, is_interactive: bool):
  """
  FN - false negative
  PFN - partially false negative
  UNC - unclear
  PTP - partially true positive
  TP - true positive
  """
  PRED_PROB_INTERVALS = {
    "FN": pd.Interval(0, 0.2),
    "PFN": pd.Interval(0.2, 0.4),
    "UNC": pd.Interval(0.4, 0.6),
    "PTP": pd.Interval(0.6, 0.8),
    "TP": pd.Interval(0.8, 1)
  }
  
  for dataset_name, dataset_results in results_list.items():
    results = pd.DataFrame(columns=['Detector', 'FN', 'PFN', 'UNC', 'PTP', 'TP'])
    results = results.astype({"Detector": "object", 
                    "FN": "float64", 
                    "PFN": "float64", 
                    "UNC": "float64", 
                    "PTP": "float64", 
                    "TP": "float64"})
    for detector in dataset_results:
      pred_data = pd.DataFrame({
        "pred_prob": detector["machine_prob"]["test"], 
        "true_label": detector["input_data"]["test"]["label"]})
      pred_data_positive = pred_data[pred_data["true_label"] == 1]
      pred_data_positive["confusion_label"] = pred_data_positive.apply(lambda row: _find_interval_label(PRED_PROB_INTERVALS, row["pred_prob"]), axis="columns")
      
      counts = dict(Counter(list(pred_data_positive["confusion_label"])))
      counts_sum = sum(counts.values())
      if counts_sum == 0:
        print("Cannot perform false negatives analysis on data with no positive samples. Exiting...")
        return
      
      results = pd.concat([results, pd.DataFrame(
        {"Detector": detector["name"], 
         "FN": counts.get("FN", 0) / counts_sum * 100,
         "PFN": counts.get("PFN", 0) / counts_sum * 100,
         "UNC": counts.get("UNC", 0) / counts_sum * 100,
         "PTP": counts.get("PTP", 0) / counts_sum * 100,
         "TP": counts.get("TP", 0) / counts_sum * 100}, index=[0])], axis="index", ignore_index=True)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    results.plot(stacked=True, ax=ax, kind="barh", x="Detector")
    ax.set(title=f"{dataset_name}/False Negatives Analysis", xlim=(0,100))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    plt.savefig(os.path.join(save_path, f"{dataset_name}_false_negatives_analysis.png"), dpi=600)
    if is_interactive:
      plt.show()

#########################################
#                  Main                 #
#########################################


FULL_ANALYSIS=[globals()[key] for key in globals() if key.startswith("analyze")]

sns.set() 

def run_full_analysis(results, methods, save_path, is_interactive: bool):  
  method_names = list(map(lambda method: method["name"], methods))
  results = make_method_names_unique(results)
  for fn in FULL_ANALYSIS:
    if fn.__name__ not in method_names and method_names[0] != "all":
      continue
    try:
      fn(results, save_path, is_interactive)
    except Exception:
      print(f"Analysis with the function {fn.__name__} failed due to below reasons. Skipping and continuing with the next function.")
      print(traceback.format_exc())
    

def run_full_analysis_from_file(filepath: str, save_path: str):
  with open(filepath, "r") as file:
    results = json.load(file, parse_float=True)
    run_full_analysis(results, save_path, is_interactive=True)


def list_available_analysis_methods():
  print("analyze_test_metrics     ...   Separate barplot for each tested metric (Accuracy, Precision, Recall, F1 score) comparing the performance of different methods")
  print("analyze_text_lengths     ...   Barplot and a lineplot of F1 score evaluated on different text lengths")
  print("pred_prob_hist           ...   For each method evaluated on a given dataset, show a histogram of prediction probability values (probability that a text is machine-generated)")
  print("pred_prob_error_hist     ...   For each method evaluated on a given dataset, show a histogram of errors in prediction (how far was the prediction from true label, how often)")
  print("analyze_false_positives  ...   Analyze false positive rates")
  print("analyze_false_negatives  ...   Analyze false negative rates")

def make_method_names_unique(results):
  for _, dataset in results.items():
    name_counts = dict()
    for method in dataset:
      count = name_counts.get(method["name"], 0)
      name_counts[method["name"]] = count + 1
      if name_counts[method["name"]] > 1:
        method["name"] = method["name"] + str(name_counts[method["name"]])
  return results

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("Please, specify file to run the analysis on and a save path to store analysis results. Aborting...")
    exit(1)
  run_full_analysis_from_file(sys.args[1], sys.args[2])
  