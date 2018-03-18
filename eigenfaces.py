import sys
from random import choice
from numpy import array, mean, zeros, sum, fromstring, full, vectorize, argsort
from numpy.linalg import eig, norm
from pandas import read_csv
from matplotlib import pyplot as plt
from time import time


def get_training_and_test_data():
    """Read and reformat traning data."""
    data = read_csv("training.csv", sep=',', header='infer')

    images = zeros((7049, 9216))
    for index, img in enumerate(data["Image"]):
        images[index] = fromstring(img, dtype=int, sep=" ")

    return (choose_training_images(images),
            choose_training_images(images, cardinality=100))


def dimensionality_reduction(matrix, reduction_filter=full((3, 3), 1 / 9),
                             image_shape=(96, 96)):
    """Reduce dimension of a image."""
    matrix = matrix.reshape(image_shape)
    (x_dim, y_dim) = reduction_filter.shape
    (mx_dim, my_dim) = matrix.shape

    x_new = int(mx_dim / x_dim)
    y_new = int(my_dim / y_dim)

    dim_reduction = zeros((x_new, y_new))
    for x in range(x_new):
        for y in range(y_new):

            x_from = x * x_dim
            x_to = x_from + x_dim

            y_from = y * y_dim
            y_to = y_from + y_dim

            dim_reduction[x, y] =\
                sum(reduction_filter * matrix[x_from:x_to, y_from:y_to])

    return dim_reduction.flatten()


def choose_training_images(images, cardinality=500):
    """
    Choose a random number of images.

    This is currently flawed because it might
    choose duplicates at the moment.
    """
    training_images = zeros((cardinality, 32 * 32))

    for index in range(cardinality):
        training_images[index] =\
            dimensionality_reduction(choice(images))

    return training_images


def vector_to_image(vector, image_shape):
    """Imagize a vector."""
    im_dim = vector.reshape(image_shape)
    image = zeros((image_shape[0], image_shape[1], 3))

    kind_of_abs = lambda x: x if x > 0 else 255
    for xi, row in enumerate(im_dim):
        for yi, cell in enumerate(row):
            image[xi, yi] = [kind_of_abs(im_dim[xi, yi]) / 256] * 3

    return image


class Eigenclassifier(object):
    """Classifier scope."""

    def __init__(self, base_dim=5, threshold=lambda x: x, verbose=True):
        """Constructor."""
        self.base_dim = base_dim
        self.threshold = threshold
        self.eigen_base = None
        self.mean_face = None
        self.covariance = None
        self.expected_values = None
        self.verbose = verbose

    def show_principal_component(self, shape=(32, 32)):
        """Show the eigen face of the principal component."""
        if self.eigen_base is None:
            sys.stderr.write(
                "Must build base before trying to show the classifier")
            sys.exit(1)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

        principal_component_0 = self.eigen_base[0]
        principal_component_1 = self.eigen_base[1]
        principal_component_2 = self.eigen_base[2]
        principal_component_3 = self.eigen_base[3]

        project_into_origin_space =\
            lambda eigen: self.expected_values.T @ eigen\
            + self.mean_face

        pc_project_0 = project_into_origin_space(principal_component_0)
        pc_project_1 = project_into_origin_space(principal_component_1)
        pc_project_2 = project_into_origin_space(principal_component_2)
        pc_project_3 = project_into_origin_space(principal_component_3)

        ax1.imshow(vector_to_image(pc_project_0, shape))
        ax2.imshow(vector_to_image(pc_project_1, shape))
        ax3.imshow(vector_to_image(pc_project_2, shape))
        ax4.imshow(vector_to_image(pc_project_3, shape))

        plt.tight_layout()
        plt.show()

    def build_base(self, training_data):
        """
        Build the eigen base.

        Expected value of any image will be assumed
        to be the mean image, this covariance is calculated
        in the following way:

        cov(X, Y) = (X - M)(Y - M)

        where M is the mean vector
        """
        time_stamp_prebuild = time()
        self.mean_face = mean(training_data, axis=0)

        """ This broadcasts the mean_vector through the matrix """
        self.expected_values = training_data - self.mean_face
        self.covariance = self.expected_values @ self.expected_values.T

        (_, y_dim) = self.covariance.shape
        (scalars, vectors) = eig(self.covariance)

        if len(scalars) < self.base_dim:
            print("Cannot build eigen base of " + self.base_dim
                  + " size, not enough eigen values exist "
                  + str(len(scalars)) + " eigen values were found")

            sys.exit(1)

        im_stripper = vectorize(lambda x: x.real)
        max_eig_index = argsort(scalars)[-self.base_dim:]
        self.eigen_base = zeros((self.base_dim, y_dim))
        for index, eigen in enumerate(max_eig_index):
            scalar = scalars[eigen]
            vector = vectors[eigen]

            if abs(scalar.imag) > 0.00001:
                print("You've done goofed, yer eigen values "
                      + "have huuuge imaginary parts, they're this: "
                      + str(scalar) + " Huuuge, go rethink your life "
                      + "choices")

                sys.exit(1)

            im_stripped = im_stripper(vector)
            self.eigen_base[index] = im_stripped / norm(im_stripped)

        """
        Perhaps there's something to gain by orthonormalizing
        the base? To do that apply repeated projection and subtraction
        i might implement it later if noone else does.
        """

        if self.verbose:
            print(
                "base took {} seconds to build".format(
                    time() - time_stamp_prebuild
                ))

        return self.eigen_base

    def predict(self, face):
        """
        Predict on new face.

        The face is assumed to already be reshaped
        into the shape that were used for traning.
        """
        if self.eigen_base is None:
            print("Build base before prediction dummy")

            sys.exit(0)

        """Calculate expected face value."""
        face = face - self.mean_face

        base_projection = zeros(self.base_dim)
        for index, base_vector in enumerate(self.eigen_base):
            vector = zeros(self.base_dim)
            vector[index] = base_vector @ face

            base_projection = base_projection + vector

        """
        Now something has to be done with the projection
        ...
        """

        return base_projection


time_stamp_pre_datafetch = time()
(training_data, test_data) = get_training_and_test_data()
print("took {} seconds to read data".format(time() - time_stamp_pre_datafetch))



classifier = Eigenclassifier()
base = classifier.build_base(training_data)
classifier.show_principal_component()




# print(images[0])
# print(len(images[0]))
# print(array(data[0])[0])
# print("HI")
# print(classifier.build_base(data))

