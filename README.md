# ML-for-CS-Lab-2

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2
   
## Folder Structure   
```bash
├── data
    └── bd_test.h5  // this is poisoned test data
    └── bd_valid.h5 // poisoned validation data
    └── data.txt  // instructions to download data
    └── test.h5  // clean test data
    └── valid.h5 // clean validation data
├── models
    └── bd_net.h5 // backdoored neural network classifier with N classes
    └── bd_net_prime_2_percent_threshold // repaired network at 2% threshold used with "bd_net.h5"
    └── bd_net_prime_4_percent_threshold // repaired network at 4% threshold used with "bd_net.h5"
    └── bd_net_prime_10_percent_threshold // repaired network at 10% threshold used with "bd_net.h5"
    └── bd_net_prime_22_percent_threshold // repaired network at 22% threshold used with "bd_net.h5"
    └── bd_weights.h5 // weights for the backdoored neural network classifier "bd_net.h5"
├── architecture.py // DNN architecture
└── Lab2_report // A short report with model accuracy and ASR
└── performanc.csv // data recorded during pruning
└── plot.png // model accuracy and ASR plot against fraction neurons pruned
└── README.md // 
└── eval.py // evaluation script
```
