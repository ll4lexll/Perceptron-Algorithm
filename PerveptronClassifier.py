import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (320845274,)

        self.weights = None
        self.classes = None

    # def fit(self, X: np.ndarray, y: np.ndarray):
    #     """
    #     This method trains a multiclass perceptron classifier on a given training set X with label set y.
    #     :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
    #     Array datatype is guaranteed to be np.float32.
    #     :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
    #     Array datatype is guaranteed to be np.uint8.
    #     """
    #
    #     # TODO - your code here
    #     # init operations
    #     self.classes = len(np.unique(y))
    #     self.weights = np.zeros((self.classes, X.shape[1] + 1)) # with bias weight
    #
    #     for cls_index in range(self.classes):
    #         wrong_found = False
    #         for x_index, x in enumerate(X):
    #             x_attr_w_bias = np.append([1], x)
    #             y_decision_val = (1 if y[x_index] == cls_index else -1)
    #             if np.dot(x_attr_w_bias, self.weights[cls_index]) * y_decision_val <= 0:
    #                 wrong_found = True
    #                 wrong_x = x_attr_w_bias
    #                 break
    #         while wrong_found:
    #             self.weights[cls_index] += wrong_x * y_decision_val
    #             wrong_found = False
    #             for x_index, x in enumerate(X):
    #                 x_attr_w_bias = np.append([1], wrong_x)
    #                 y_decision_val = (1 if y[x_index] == cls_index else -1)
    #                 if np.dot(x_attr_w_bias, self.weights[cls_index]) * y_decision_val <= 0:
    #                     wrong_found = True
    #                     wrong_x = x_attr_w_bias
    #                     break


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """

        # TODO - your code here
        # init operations
        self.classes = len(np.unique(y))
        self.weights = np.zeros((self.classes, X.shape[1] + 1)) # with bias weight

        to_continue = True
        while to_continue:
            to_continue = False

            for x_idx, x_vec in enumerate(X):
                x_with_one = np.append([1], x_vec)
                in_products = np.array([[np.dot(x_with_one, self.weights[cls]),
                                         cls] for cls in range(self.classes)])
                sorted_prod = in_products[in_products[:, 0].argsort()]
                y_prediction = sorted_prod[-1, -1]
                y_true = y[x_idx]
                if y_prediction != y_true:
                    self.weights[y_true] = self.weights[y_true] + x_with_one
                    self.weights[(int)(y_prediction)] = self.weights[(int)(y_prediction)] - x_with_one
                    to_continue = True





    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        # TODO - your code here
        predictions = []
        for x_idx, x_vec in enumerate(X):
            x_with_one = np.append([1], x_vec)
            in_products = np.array([[np.dot(x_with_one, self.weights[cls]),
                                     cls] for cls in range(self.classes)])
            sorted_prod = in_products[in_products[:, 0].argsort()]
            y_prediction = sorted_prod[-1, -1]
            predictions.append(y_prediction)
        return np.array(predictions)

        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


if __name__ == "__main__":

    print("*" * 20)
    print("Started HW2_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
