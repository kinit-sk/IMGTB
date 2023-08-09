import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from math import ceil, sqrt
import os
import traceback
import sys
import json


def analyze_test_metrics(results_list, save_path):
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
    plt.xticks(rotation=25)
    
    fig.add_subplot(rows, cols, 2)
    ax = sns.barplot(results, x="Detector", y="Accuracy")
    ax.set(ylim=(0,1))
    plt.xticks(rotation=25)

    fig.add_subplot(rows, cols, 3)
    ax = sns.barplot(results, x="Detector", y="Precision")
    ax.set(ylim=(0,1))
    plt.xticks(rotation=25)
    
    fig.add_subplot(rows, cols, 4)
    ax = sns.barplot(results, x="Detector", y="Recall")
    ax.set(ylim=(0,1))
    plt.xticks(rotation=25)

    plt.subplots_adjust(left=0.1,
                      bottom=0.155,
                      right=0.9,
                      top=0.85,
                      wspace=0.33,
                      hspace=0.8)
  
    plt.savefig(os.path.join(save_path, f"{dataset_name}_metrics_analysis.png"))
    plt.show()



def analyze_text_lengths(results_list, save_path):
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
        if len(temp.label.unique()) < 2: continue
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
      
    plt.savefig(os.path.join(save_path, f'{dataset_name}_text_lengths_analysis.png'))
    plt.show()


FULL_ANALYSIS=[analyze_test_metrics, analyze_text_lengths]
sns.set() 

def run_full_analysis(results, save_path):
  for fn in FULL_ANALYSIS:
    try:
      fn(results, save_path)
    except Exception:
      print(f"Analysis with the function {fn.__name__} failed due to below reasons. Skipping and continuing with the next function.")
      print(traceback.format_exc())
    

def run_full_analysis_from_file(filepath: str, save_path: str):
  with open(filepath, "r") as file:
    results = json.load(file, parse_float=True)
    run_full_analysis(results, save_path)
    

if __name__ == '__main__':
  if sys.argv != 3:
    print("Please, specify file to run the analysis on and a save path to store analysis results. Aborting...")
    exit(1)
  run_full_analysis_from_file(sys.args[1], sys.args[2])