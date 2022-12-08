# ML-for-CS-Lab-2

This repository implements a pruning defense on a backdoored face recognition nueral network as described in this paper:

[Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://arxiv.org/abs/1805.12185)

Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg

## Contents
- [Dependencies](#Dependencies)
   - [Libraries](#Libraries)
   - [Dataset](#Dataset)
- [Usage](#Usage)
   - [Running the program from Command Line Interface](#Running-the-program-from-Command-Line-Interface)
   - [Running the program from aother Python file](#Running-the-program-from-aother-Python-file)
- [Arguments](#Arguments)
- [Architecture](#Architecture)
- [Folder Structor](#Folder-Structure)

# Dependencies
## Libraries
* Python 3.6.9
* Keras 2.3.1
* Numpy 1.16.3
* Matplotlib 2.2.2
* H5py 2.9.0
* TensorFlow-gpu 1.15.2

## Dataset
Download the validation and test datasets from [here](https://drive.google.com/drive/folders/1jjcdL20CHWmAd9DsIdgP_B7IIEARKdSx) and store them under the [data/](data/) directory.

To run the program with your own data files instead, you can either provide the paths of the data files as arguments as explained in the [Usage](#Usage) section. 

Alternatively, you can run the program with default arguments by adding your files in the [data/](data/) directory with this naming scheme:
- "bd_test.h5" - poisoned test data
- "test.h5" - clean test data
- "valid.h5" - clean validation data

# Usage
The program can be run from [CLI](#Running-the-program-from-Command-Line-Interface) or by [standard imports](#Running-the-program-from-aother-Python-file). The default and custom arguments for the program are given in the [Arguments](#Arguments) section.

## Running the program from Command Line Interface
You can run the program with the default arguments using this command:
```
$ python backdoor_detector.py
```
You can also provide the program custom arguments:
```
$ python backdoor_detector.py --model "model/bd_net.h5" --Dvalid --Dtest --Btest --thresholds
```
   
## Running the program from aother Python file

## Arguments

# Architecture

## Folder Structure   
```bash
├── data
    └── bd_test.h5  # this is poisoned test data
    └── bd_valid.h5 # poisoned validation data
    └── data.txt  # instructions to download data
    └── test.h5  # clean test data
    └── valid.h5 # clean validation data
├── models
    └── bd_net.h5 # backdoored neural network classifier with N classes
    └── bd_net_prime_2_percent_threshold # repaired network at 2% threshold used with "bd_net.h5"
    └── bd_net_prime_4_percent_threshold # repaired network at 4% threshold used with "bd_net.h5"
    └── bd_net_prime_10_percent_threshold # repaired network at 10% threshold used with "bd_net.h5"
    └── bd_net_prime_22_percent_threshold # repaired network at 22% threshold used with "bd_net.h5"
    └── bd_weights.h5 # weights for the backdoored neural network classifier "bd_net.h5"
├── architecture.py # DNN architecture
└── backdoor_detector.py # program the implements the pruning defense on backdoored network
└── eval.py # evaluation script
└── Lab2_report.docx # A short report with model accuracy and ASR
└── performanc.csv # data recorded during pruning
└── plot.png # model accuracy and ASR plot against fraction neurons pruned
└── README.md # 
└── utils.py # Utilty methods and class for ensemble repaired network
```
