from numpy import array, mean
from numpy.linalg import eig
from pandas import read_csv
import csv


class Eigenclassifier(object):
    """Classifier scope."""

    def __init__(self, base_dim=5, threshold=lambda x: x):
        """Constructor."""
        self.base_dim = base_dim
        self.threshold = threshold
        self.eigen_base = None

    def build_base(self, training_data):
        """
        Build the eigen base.

        Expected value of any image will be assumed
        to be the mean image, this covariance is calculated
        in the following way:

        cov(X, Y) = (X - M)(Y - M)

        where M is the mean vector
        """
        mean_vector = mean(training_data, axis=0)

        """ This broadcasts the mean_vector through the matrix """
        expected_values = training_data - mean_vector

        """
        The article says to calculate the covariance matrix
        but i don't understand why or where it is used.
        """
        # covariance_matrix = expected_values @ expected_values.T

        # basis = expected_values.T @ expected_values

        # eigen_stuff = eig(basis)

        return expected_values





data = read_csv("training.csv", sep=',', header='infer')
data = data["Image"]

classifier = Eigenclassifier()

print(data[0])
print(len(data[0]))
print("HI")
print(classifier.build_base(data))

