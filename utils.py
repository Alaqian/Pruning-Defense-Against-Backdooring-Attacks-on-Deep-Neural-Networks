import numpy as np
import tensorflow as tf


class G(tf.keras.Model):
    def __init__(self, B, B_prime):
        super(G, self).__init__()
        self.B = B
        self.B_prime = B_prime

    def predict(self, data):
        y_pred = self.B(data)
        y = np.argmax(y_pred, axis=1)
        y_prime = np.argmax(self.B_prime(data), axis=1)
        res = np.zeros((y.shape[0], 1284))
        for i in range(y.shape[0]):
            if y[i] == y_prime[i]:
                res[i, :-1] = y_pred[i, :]
            else:
                res[i, 1283] = 1
        return res

    # For small amount of inputs that fit in one batch, directly using call() is recommended for faster execution,
    # e.g., model(x), or model(x, training=False) is faster then model.predict(x) and do not result in
    # memory leaks (see for more details https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)
    def call(self, data):
        y_pred = self.B(data)
        y = np.argmax(y_pred, axis=1)
        y_prime = np.argmax(self.B_prime(data), axis=1)
        res = np.zeros((y.shape[0], 1284))
        for i in range(y.shape[0]):
            if y[i] == y_prime[i]:
                res[i, :-1] = y_pred[i, :]
            else:
                res[i, 1283] = 1
        return res


def evaluate_model(bd_model, cl_x_test, cl_y_test, bd_x_test, bd_y_test):
    """
    Evaluates the performance of a given model on clean and backdoored test data.

    Args:
        bd_model: The model to evaluate.
        cl_x_test: The clean test input data.
        cl_y_test: The clean test labels.
        bd_x_test: The backdoored test input data.
        bd_y_test: The backdoored test labels.

    Returns:
        A tuple containing the model's clean accuracy and ASR.
    """

    clean_accuracy = calculate_model_accuracy(bd_model, cl_x_test, cl_y_test)
    asr = calculate_model_asr(bd_model, bd_x_test, bd_y_test)
    return clean_accuracy, asr


def calculate_model_accuracy(bd_model, cl_x_test, cl_y_test):
    """
    Calculates the accuracy of a given model on clean test data.

    Args:
        bd_model: The model to evaluate.
        cl_x_test: The clean test input data.
        cl_y_test: The clean test labels.

    Returns:
        The model's clean accuracy.
    """

    cl_label_p = np.argmax(bd_model(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100
    return clean_accuracy


def calculate_model_asr(bd_model, bd_x_test, bd_y_test):
    """
    Calculates the ASR (average success rate) of a given model on backdoored test data.

    Args:
        bd_model: The model to evaluate.
        bd_x_test: The backdoored test input data.
        bd_y_test: The backdoored test labels.

    Returns:
        The model's ASR.
    """

    bd_label_p = np.argmax(bd_model(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
    return asr
