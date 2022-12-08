# ML-for-CS-Lab-2

This repository implements a pruning defense on a backdoored face recognition nueral network as described in this paper:

[Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://arxiv.org/abs/1805.12185)

Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg

- [Dependencies](#Dependencies)
- [Folder Structor](#Folder-Structure)

## Dependencies
* Python 3.6.9
* Keras 2.3.1
* Numpy 1.16.3
* Matplotlib 2.2.2
* H5py 2.9.0
* TensorFlow-gpu 1.15.2
   
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
└── Lab2_report # A short report with model accuracy and ASR
└── performanc.csv # data recorded during pruning
└── plot.png # model accuracy and ASR plot against fraction neurons pruned
└── README.md # 
└── eval.py # evaluation script
```
