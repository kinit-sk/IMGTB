import os

if __name__ == "__main__":
    os.system("python benchmark.py --from_config tests/testing_config.yaml")
    os.system("python benchmark.py -i --name only_test_to_be_removed --dataset tests/datasets/test_mix_labels.csv --dataset tests/datasets/test_machine_only --methods all EntropyMetric DetectGPT --clf_algo_for_threshold RandomForestClassifier --base_model_name prajjwal1/bert-tiny --mask_filling_model_name google/t5-efficient-tiny --analysis_methods all")
    os.system("python benchmark.py --list_datasets")
    os.system("python benchmark.py --list_methods")
    os.system("python benchmark.py --list_analysis_methods")