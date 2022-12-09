import keras
import sys
import h5py
import numpy as np
from utils import *

"""
- eval.py uses just the model provided:
model = B
prediction = model(x_data)

- eval_ensemble.py uses B - The backdoored network and B_prime - The repaired network:
model = G(B, B_prime)
prediction = model(x_data)
"""

def data_loader(filepath):
    data = h5py.File(filepath, "r")
    x_data = np.array(data["data"])
    y_data = np.array(data["label"])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def main():
    clean_data_filename = str(sys.argv[1])
    poisoned_data_filename = str(sys.argv[2])
    model_filename = str(sys.argv[3])
    model_prime = str(sys.rgv[4])

    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    model = keras.models.load_model(model_filename)
    model_prime = keras.models.load_model(model_prime)
    bd_model = G(model, model_prime)

    cl_label_p = np.argmax(bd_model(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    print("Clean Classification accuracy:", clean_accuracy)

    bd_label_p = np.argmax(bd_model(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    print("Attack Success Rate:", asr)


if __name__ == "__main__":
    main()
