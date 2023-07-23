# MGTBench

MGTBench provides the reference implementations of different machine-generated text (MGT) detection methods.
It is still under continuous development and we will include more detection methods as well as analysis tools in the future.


## Supported Methods
Currently, we support the following methods (continuous updating):
- Metric-based methods:
    - Log-Likelihood [[Ref]](https://arxiv.org/abs/1908.09203);
    - Rank [[Ref]](https://arxiv.org/abs/1906.04043);
    - Log-Rank [[Ref]](https://arxiv.org/abs/2301.11305);
    - Entropy [[Ref]](https://arxiv.org/abs/1906.04043);
    - GLTR Test 2 Features (Rank Counting) [[Ref]](https://arxiv.org/abs/1906.04043);
    - DetectGPT [[Ref]](https://arxiv.org/abs/2301.11305);
- Model-based methods:
    - OpenAI Detector [[Ref]](https://arxiv.org/abs/1908.09203);
    - ChatGPT Detector [[Ref]](https://arxiv.org/abs/2301.07597);
    - GPTZero [[Ref]](https://gptzero.me/);
    - LM Detector [[Ref]](https://arxiv.org/abs/1911.00650);

## Supported Datasets
- TruthfulQA;
- SQuAD1;
- NarrativeQA; 

For datasets, you can download them from [Google Drive](https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola?usp=sharing).

## Installation
```
git clone https://github.com/xinleihe/MGTBench.git;
cd MGTBench;
conda env create -f environment.yml;
conda activate MGTBench;
```

## Usage
To run the benchmark on the SQuAD1 dataset: 
```
# Distinguish Human vs. ChatGPT:
python benchmark.py --dataset SQuAD1 --detectLLM ChatGPT

# Text attribution:
python attribution_benchmark.py --dataset SQuAD1

Note that you can also specify your own datasets on ``dataset_loader.py``.

## Authors
The tool is designed and developed by Xinlei He (CISPA), Xinyue Shen (CISPA), Zeyuan Chen (Individual Researcher), Michael Backes (CISPA), and Yang Zhang (CISPA).

## Cite
If you use MGTBench for your research, please cite [MGTBench: Benchmarking Machine-Generated Text Detection](https://arxiv.org/abs/2303.14822).

```bibtex
@article{HSCBZ23,
author = {Xinlei He and Xinyue Shen and Zeyuan Chen and Michael Backes and Yang Zhang},
title = {{MGTBench: Benchmarking Machine-Generated Text Detection}},
journal = {{CoRR abs/2303.14822}},
year = {2023}
}
```

# Integrated MGTBench Framework
A framework, heavily based upon the original tool - [MGTBench: Benchmarking Machine-Generated Text Detection](https://arxiv.org/abs/2303.14822) - with additional funcionalities that make it easier to integrate custom datasets and custom detection methods

## Support for custom dataset integration
In the CLI you can select multiple dataset files (these have to be identified by a filepath) to be read from. In the CLI you also have to define a processor which will be a function that will process your selected dataset files into a unified data format.

The processor is a function defined as follows:

Name: process_PROCESSOR-NAME (Here, PROCESSOR-NAME will be the selected name of your processor. This would be usually the name of the dataset)
Input: 2 arguments: list of pandas dataframes for each dataset file, list of strings corresponding to the --dataset_other command-line argument 
Output: a tuple of 2 lists with human and machine texts correspondingely

You will have to define this function in the dataset_loader.py file.

Examples usage could be:

python benchmark.py --dataset_filepaths datasets/human_texts.csv datasets/machine_texts.json --dataset_processor myAwesomeDataset

## Support for custom MGTD method integration

To integrate a new method, you need to define new `Experiment` subclass in the `methods/implemented_methods directory`. The main script in `benchmark.py` will automatically detect (unless you choose otherwise by configuring the `--methods` option) your new method and evaluate it on your chosen dataset.

### How to implement a new Experiment subclass

To implement a new method, you can use one of the templates in the `methods/method_templates`. You will just have to fill in the not yet implemented methods and maybe tweak the `__init__()` constructor.

### Experiment constructor parameters

To correctly setup the input parameters in `__init__()` you will have to have a look at this line in `benchmark.py`:

```python
    outputs = list(map(lambda obj: obj(data=data, 
                                       model=base_model, 
                                       tokenizer=base_tokenizer, 
                                       DEVICE=DEVICE, 
                                       detectLLM=args.detectLLM, 
                                       batch_size=batch_size,
                                       cache_dir=cache_dir,
                                       args=args,
                                       gptzero_key=args.gptzero_key
                                       ).run(), filtered))
```

Each `Experiment` object is initialized with these parameters. We only use keyword (named) parameters, keep that in mind while naming your parameters in the `__init__()` constructor. 
Optionally, we use `**kwargs` int eh `__init__()` parameters to catch remaining (unused) parameters.

#### Note:
While developing your new method, you might find useful some of the functionality in `methods/utils.py`

## Authors
The framework was built upon the original MGTBench tool, designed and developed by Michal Spiegel (KINIT) under the supervision of Dominik Macko (KINIT)