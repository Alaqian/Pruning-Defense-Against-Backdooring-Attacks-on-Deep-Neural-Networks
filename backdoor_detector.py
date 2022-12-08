from eval import data_loader
import keras
from keras.models import Model
from utils import *
import argparse
import warnings
import matplotlib.pyplot as plt
import csv

warnings.filterwarnings("ignore")


def main():
    args = _parse_arguments()
    B_path = args.model  # Path of the backdoored nueral network classifier
    Dvalid = args.Dvalid  # Path of the clean validation data
    Dtest = args.Dtest  # Path of the clean test data
    Btest = args.Btest  # Path of the poisoned test data
    thresholds = args.thresholds  # threshold percentage

    Goodnets = backdoor_detector(
        B_path=B_path, Dvalid=Dvalid, Dtest=Dtest, Btest=Btest, thresholds=thresholds
    )


def backdoor_detector(B_path=None, Dvalid=None, Dtest=None, Btest=None, thresholds=[2, 4, 10]):

    """
    Repairs a backdoor in a deep learning model by pruning channels in the second-last convolutional
    layer for thresholds of drop in accuracy on clean validation data. Saves the prime models on to the
    disk in the same folder as the badnets.

    Args:
    B_path: str. The path to the model (in HDF5 format).
    Dvalid: str. The path to the validation data (in HDF5 format).
    Dtest: str. The path to the clean data (in HDF5 format).
    Btest: str. The path to the poisoned data (in HDF5 format).
    thresholds: list. A list of accuracy thresholds in percent.

    Returns:
    A list of the repaired Goodnet models.
    """

    if B_path == None:
        B_path = "model/bd_net.h5"

    if Dvalid == None:
        Dvalid = "data/valid.h5"

    if Dtest == None:
        Dtest = "data/test.h5"

    if Btest == None:
        Btest = "data/bd_test.h5"

    cl_x_valid, cl_y_valid = data_loader(Dvalid)  # load the clean validation data
    cl_x_test, cl_y_test = data_loader(Dtest)  # load the clean test data
    bd_x_test, bd_y_test = data_loader(Btest)  # load the poisoned test data
    B = keras.models.load_model(B_path)  # load the model
    B_clone = keras.models.load_model(B_path)  # load a clone of the model
    clean_accuracy = calculate_model_accuracy(B, cl_x_valid, cl_y_valid)  # get clean accuracy
    model_performance = []  # initialize model performance
    test_accuracy, test_asr = evaluate_model(B, cl_x_test, cl_y_test, bd_x_test, bd_y_test)
    model_performance.append((0, test_accuracy, test_asr))

    # Redefine model to output right after the last pooling layer ("pool_3")
    intermediate_model = Model(inputs=B.inputs, outputs=B.get_layer("pool_3").output)

    # Get feature map for last pooling layer ("pool_3") using the clean validation data and intermediate_model
    feature_maps_cl = intermediate_model(cl_x_valid)

    # Get average activation value of each channel in last pooling layer ("pool_3")
    averageActivationsCl = np.mean(feature_maps_cl, axis=(0, 1, 2))

    # Store the indices of average activation values (averageActivationsCl) in increasing order
    idxToPrune = np.argsort(averageActivationsCl)

    # Get the conv_4 layer weights and biases from the original network that will be used for prunning
    lastConvLayerWeights = B.get_layer("conv_3").get_weights()[0]
    lastConvLayerBiases = B.get_layer("conv_3").get_weights()[1]

    # Sort the thresholds in ascending order
    thresholds = sorted(thresholds)

    thresholds.append(101)

    # Initialize the list of repaired models
    goodnets = []

    # Initialize the index for the current threshold
    i = 0

    # Initialize the index for the current threshold
    threshold = thresholds[i]

    print("beginning to prune the network...")
    for j, chIdx in enumerate(idxToPrune):
        # Prune one channel at a time
        lastConvLayerWeights[:, :, :, chIdx] = 0
        lastConvLayerBiases[chIdx] = 0

        # Update weights and biases of B_clone
        B_clone.get_layer("conv_3").set_weights([lastConvLayerWeights, lastConvLayerBiases])

        # Evaluate the updated model's (B_clone) clean validation accuracy
        clean_accuracy_valid = calculate_model_accuracy(B_clone, cl_x_valid, cl_y_valid)
        repaired_net = G(B, B_clone)
        test_accuracy, test_asr = evaluate_model(
            repaired_net, cl_x_test, cl_y_test, bd_x_test, bd_y_test
        )
        model_performance.append((j + 1, test_accuracy, test_asr))
        print(f"{j + 1} neurons removed, test_accuracy: {test_accuracy:.3f}% ASR: {test_asr:.3f}%")

        # If drop in clean_accuracy_valid is just greater (or equal to) than the desired threshold
        # compared to clean_accuracy, then save B_clone as B_prime and break
        if clean_accuracy - clean_accuracy_valid >= threshold:
            # Save B_clone to disk and return it from the function
            model_filename = f"{B_path[:-3]}_prime_{threshold}_percent_threshold.h5"
            B_clone.save(model_filename)
            print(f"Saving repaired network for {threshold}% threshold at: {model_filename}")
            goodnets.append(repaired_net)
            i += 1
            threshold = thresholds[i]

    model_performance = np.array(model_performance)
    _save_model_performance_plot(model_performance)
    _save_model_performance_data(model_performance)

    return goodnets


def _save_model_performance_plot(model_performance):
    # Calculate the fraction of neurons pruned
    total_nodes = model_performance[-1, 0]
    fraction_nodes_pruned = model_performance[:, 0] / total_nodes

    # Create a figure and set its size
    fig = plt.figure(figsize=(8, 6))

    # Plot the performance on the Axes object
    plt.plot(fraction_nodes_pruned, model_performance[:, 1], label="Clean Classification Accuracy")
    plt.plot(fraction_nodes_pruned, model_performance[:, 2], label="Backdoor Attack Success")

    # Add labels and titles to the Axes object
    plt.xlabel("Fraction of Neurons Pruned")
    plt.ylabel("Rate")
    plt.title("Model accuracy and ASR vs fraction of nodes pruned")

    # Add a legend
    plt.legend()

    # Save the figure as an image file
    fig.savefig("plot.png")


def _save_model_performance_data(model_performance):
    # Define the headings
    headings = ["Neurons Pruned", "Accuracy", "ASR"]

    # Save the performance to a CSV file
    with open("performance.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headings)
        writer.writerows(model_performance)


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--model",
        help="Enter path to the model with weights in .h5 format",
        type=str,
        default=None,
    )
    argparser.add_argument(
        "-v",
        "--Dvalid",
        help="Enter path to the clean validation data in .h5 format",
        type=str,
        default=None,
    )
    argparser.add_argument(
        "-t",
        "--Dtest",
        help="Enter path to the clean test data in .h5 format",
        type=str,
        default=None,
    )
    argparser.add_argument(
        "-b",
        "--Btest",
        help="Enter path to the posisoned test data in .h5 format",
        type=str,
        default=None,
    )
    argparser.add_argument(
        "-th",
        "--thresholds",
        help="Enter a list of thresholds in percentage eg. [2,4,10] or [5]",
        type=int,
        default=[2, 4, 10],
        nargs="+",
    )
    return argparser.parse_args()


if __name__ == "__main__":
    main()
