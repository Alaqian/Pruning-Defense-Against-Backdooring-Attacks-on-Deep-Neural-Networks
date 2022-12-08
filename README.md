# ML-for-CS-Lab-2

This repository implements a pruning defense on a backdoored face recognition nueral network as described in this paper:

[Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://arxiv.org/abs/1805.12185)

Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg

## Contents
- [Dependencies](#Dependencies)
   - [Libraries](#Libraries)
   - [Dataset](#Dataset)
- [Usage](#Usage)
   - [Running the program from aother Python file](#Running-the-program-from-aother-Python-file)
   - [Running the program from Command Line Interface](#Running-the-program-from-Command-Line-Interface)
   - [Arguments](#Arguments)
- [Results](#Results)
- [Folder Structure](#Folder-Structure)

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
The program can be run by [standard imports](#Running-the-program-from-aother-Python-file) or the [CLI](#Running-the-program-from-Command-Line-Interface). The default and custom arguments for the program are given in the [Arguments](#Arguments) section.

## Running the program from aother Python file
You can run the program with default arguments:
```python
from backdoor_detector import backdoor_detector

repaired_net = backdoor_detector()
```
You can also provide the program with custom arguments:
```python
from backdoor_detector import backdoor_detector

repaired_net = backdoor_detector(B_path="model/bd_net.h5", Dvalid="data/valid.h5", Dtest="data/test.h5", Btest="data/bd_test.h5", thresholds=[2, 4, 10])
```
This program will:
* save repaired models as '.h5' files in the same directory as the backdoored models at the specified thresholds
* save performance data as 'performance.csv'
* plot model accuracy and backdoor attack success rate against the fraction of neurons removed
* Return the repaired model as a goodnet object that is an ensemble of the original backdoored network and the repaired network

The saved model can then be used to make predictions on data:
```python
predictions = repaired_net(x_data)
y_pred = np.argmax(predictions, axis=1)
```
## Running the program from Command Line Interface
You can run the program with the default arguments using this command:
```bash
$ python backdoor_detector.py
```
You can also provide the program custom arguments:
```bash
$ python backdoor_detector.py --model "model/bd_net.h5" --Dvalid "data/valid.h5" --Dtest "data/test.h5" --Btest "data/bd_test.h5" --thresholds 2 4 10
```

The program will:
* save repaired models as '.h5' files in the same directory as the backdoored models at the specified thresholds
* save performance data as 'performance.csv'
* plot model accuracy and backdoor attack success rate against the fraction of neurons removed

The saved model can be then loaded in a python file:
```python
from utils import *

B_path = "model/bd_net.h5" # path to backdoored model
B_prime = "bd_net_prime_20_percent_threshold" # path to pruned model
# load both models
B = keras.models.load_model(B_path)
B_p = keras.models.load_model(B_prime)
repaired_net = G(B, B_p)
predictions = repaired_net(x_data)
y_pred = np.argmax(predictions, axis=1)
```

## Arguments
| CLI Argument | Short hand | Method Argument | Description |
|---|---|---|---|
| `--model` | `-m` | `B_path` | path to backdoored model. Default: "model/bd_net.h5" |
| `--Dvalid` | `-v` | `Dvalid` | Path to the clean validation data. Default: "data/valid.h5" | 
| `--Dtest` | `-t` | `Dtest` | Path to the clean test data. Default: "data/test.h5" | 
| `--Btest` | `-b` | `Btest` | Path to the poisoned test data.. Default: "data/bd_test.h5" | 
| `--thresholds` | `-th` | `thresholds` | List of accuracy thresholds in percent. Default: `[2, 4, 10]`| 

## Results
![image](plot.png)

## Architecture
The baseline DNN used for face recognition is the state-ofthe-art DeepID network that contains three shared convolutional layers followed by two parallel sub-networks that feed into the last two fully connected layers.

The architecture of the model can is in the [architecture.py](architecture.py) file. A model summary and a flow diagram of the model is shown below.
```
___________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
===================================================================================
 input (InputLayer)             [(None, 55, 47, 3)]  0           []                               
                                                                                                  
 conv_1 (Conv2D)                (None, 52, 44, 20)   980         ['input[0][0]']                  
                                                                                                  
 pool_1 (MaxPooling2D)          (None, 26, 22, 20)   0           ['conv_1[0][0]']                 
                                                                                                  
 conv_2 (Conv2D)                (None, 24, 20, 40)   7240        ['pool_1[0][0]']                 
                                                                                                  
 pool_2 (MaxPooling2D)          (None, 12, 10, 40)   0           ['conv_2[0][0]']                 
                                                                                                  
 conv_3 (Conv2D)                (None, 10, 8, 60)    21660       ['pool_2[0][0]']                 
                                                                                                  
 pool_3 (MaxPooling2D)          (None, 5, 4, 60)     0           ['conv_3[0][0]']                 
                                                                                                  
 conv_4 (Conv2D)                (None, 4, 3, 80)     19280       ['pool_3[0][0]']                 
                                                                                                  
 flatten (Flatten)              (None, 1200)         0           ['pool_3[0][0]']                 
                                                                                                  
 flatten_1 (Flatten)            (None, 960)          0           ['conv_4[0][0]']                 
                                                                                                  
 fc_1 (Dense)                   (None, 160)          192160      ['flatten[0][0]']                
                                                                                                  
 fc_2 (Dense)                   (None, 160)          153760      ['flatten_1[0][0]']              
                                                                                                  
 add (Add)                      (None, 160)          0           ['fc_1[0][0]',      
                                                                  'fc_2[0][0]']                   
                                                                                                  
 activation (Activation)        (None, 160)          0           ['add[0][0]']                    
                                                                                                  
 output (Dense)                 (None, 1283)         206563      ['activation[0][0]']             
                                                                                                  
======================================================================================
Total params: 601,643
Trainable params: 601,643
Non-trainable params: 0
______________________________________________________________________________________
```
![image](img/architecture.png)
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
    └── bd_net_prime_20_percent_threshold # repaired network at 20% threshold used with "bd_net.h5"
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
