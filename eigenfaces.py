import sys
from random import choice
from numpy import array, mean, zeros, sum, fromstring, full, vectorize, argsort, sqrt
from numpy.linalg import eig, norm
from pandas import read_csv
from matplotlib import pyplot as plt
from time import time

"""True intmax does not exists in python3 but this'll do."""
INT_MAX = 1000000000


def get_training_test_and_real_test_data():
    """Read and reformat traning data."""
    data = read_csv("training.csv", sep=',', header='infer')
    test = read_csv("test.csv", sep=",", header="infer")

    images = zeros((7049, 9216))
    test_images = zeros((1783, 9216))
    for index, img in enumerate(data["Image"]):
        images[index] = fromstring(img, dtype=int, sep=" ")

    for index, img in enumerate(test["Image"]):
        test_images[index] = fromstring(img, dtype=int, sep=" ")

    return (choose_images(images),
            choose_images(images, cardinality=100),
            choose_images(test_images, cardinality=500))


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


def choose_images(images, cardinality=500):
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

    def __init__(self, base_dim=5, verbose=True, orthogonalize=False):
        """Constructor."""
        self.base_dim = base_dim
        self.eigen_base_ld = None
        self.eigen_base_hd = None
        self.mean_face = None
        self.covariance = None
        self.expected_values = None
        self.verbose = verbose
        self.orthogonalize = orthogonalize

        """Used in classification."""
        self.face_space = None
        self.maximal_allowed_distance = 0

    def displayable_vector(self, vector):
        """Shapes vector into displayable format."""
        mn = min(vector)
        mx = max(vector)

        stabiliser = mx + abs(mn)

        avoid_zero_division = stabiliser if stabiliser != 0 else 1

        vector = vector + abs(mn)
        return vector * (256 / avoid_zero_division)

    def project_on_space(self, vector):
        """Project the vector onto the space.

        Calculate expected face value
        """
        vector = vector - self.mean_face

        """
        Project the face onto the new base by summation of
        projection onto each of the base vectors
        """
        base_projection = zeros(self.base_dim)
        for index, base_vector in enumerate(self.eigen_base_hd):
            axis = zeros(self.base_dim)
            axis[index] = vector @ base_vector

            base_projection = base_projection + axis

        return base_projection

    def euclidian_distance(self, v1, v2):
        """Calculate the euclidian distance of two vectors."""
        return sqrt(sum((v1 - v2) * (v1 - v2)))

    def show_principal_component(self, shape=(32, 32)):
        """Show the eigen face of the principal component."""
        if self.eigen_base_hd is None:
            sys.stderr.write(
                "Must build base before trying to show the classifier")
            sys.exit(1)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

        principal_component_0 = self.displayable_vector(self.eigen_base_hd[0])
        principal_component_1 = self.displayable_vector(self.eigen_base_hd[1])
        principal_component_2 = self.displayable_vector(self.eigen_base_hd[2])
        principal_component_3 = self.displayable_vector(self.eigen_base_hd[3])

        ax1.imshow(vector_to_image(principal_component_0, shape))
        ax2.imshow(vector_to_image(principal_component_1, shape))
        ax3.imshow(vector_to_image(principal_component_2, shape))
        ax4.imshow(vector_to_image(principal_component_3, shape))

        plt.tight_layout()
        plt.show()

    def build_base(self, training_data):
        """
        Build the eigen base.

        Expected value of any image will be assumed
        to be the mean image, this covariance is calculated
        in the following way:

        cov(X, Y) = (X - Mv)(Y - Mv)

        where Mv is the mean vector

        M = number of samples
        N^2 = flattened image
        """
        time_stamp_prebuild = time()
        self.mean_face = mean(training_data, axis=0)

        """ This broadcasts the mean_vector through the matrix """
        self.expected_values = training_data - self.mean_face

        (M, NN) = self.expected_values.shape

        """
        Calculate the covariance matrix as an M X M matrix
        """
        self.covariance = self.expected_values @ self.expected_values.T

        (scalars, vectors) = eig(self.covariance)

        if len(scalars) < self.base_dim:
            print("Cannot build eigen base of " + self.base_dim
                  + " size, not enough eigen values exist "
                  + str(len(scalars)) + " eigen values were found")

            sys.exit(1)

        """
        stripps the vectors of the imaginary part
        the imaginary part should be insignificant
        otherwise something has been done wrong
        """
        im_stripper = vectorize(lambda x: x.real)

        """
        projects the vector into the image space
        because the eigen values is calculated from
        the M x M matrix rather than the N^2 * N^2
        matrix where M = number of samples and N^2
        is the flattened image vector. To be able
        to project a new vector onto the image space
        we have to project the eigen vectors back
        into image space. The reason for doing this
        is becaused of the assumption that M << N^2
        making it easier to calculate eigenvectors for
        M rather than N^2
        """

        def project_into_image_space(vector):
            image_space = self.expected_values.T @ vector
            return image_space / norm(image_space)

        """
        Gets the index of the #base_dim number of largest
        eigen scalars
        """
        max_eig_index = argsort(scalars)[-self.base_dim:]

        self.eigen_base_ld = zeros((self.base_dim, M))
        self.eigen_base_hd = zeros((self.base_dim, NN))
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
            self.eigen_base_ld[index] = im_stripped / norm(im_stripped)
            self.eigen_base_hd[index] =\
                project_into_image_space(self.eigen_base_ld[index])


        if self.orthogonalize:
            for i in range(self.base_dim):
                for j in range(i + 1, self.base_dim):
                    self.eigen_base_hd[i] =\
                        self.eigen_base_hd[i] -\
                        self.eigen_base_hd[j] @ self.eigen_base_hd[i]\
                        * self.eigen_base_hd[j]

                    self.eigen_base_ld[i] =\
                        self.eigen_base_ld[i] -\
                        self.eigen_base_ld[j] @ self.eigen_base_ld[i]\
                        * self.eigen_base_ld[j]

                    self.eigen_base_hd[i] = self.eigen_base_hd[i]\
                        / norm(self.eigen_base_hd[i])

                    self.eigen_base_ld[i] = self.eigen_base_ld[i]\
                        / norm(self.eigen_base_ld[i])

        if self.verbose:
            print(
                "base took {} seconds to build".format(
                    time() - time_stamp_prebuild
                ))

        return self.eigen_base_hd

    def get_min_distance_to_faces(self, faces, training_faces):
        """
        Define face space within the eigenbase of faces.

        It's recommended that the faces used is the faces
        that were used for building the base, but for science
        this functions allowes that to be changed to observe results!
        """
        if self.eigen_base_hd is None:
            print("Build base before space dummy")

            sys.exit(0)

        """Reset maximal_allowed distance."""
        self.maximal_allowed_distance = 0

        """
        Face space is defined as a minimal distance
        to some particular face in faces projected
        onto the eigenface_base

        Projects all faces onto the face space.
        """
        self.face_space = zeros((len(faces), self.base_dim))
        for index, face in enumerate(faces):
            self.face_space[index] = self.project_on_space(face)

        """Project all training faces onto face space."""
        _training_faces = zeros((len(training_faces), self.base_dim))
        for index, face in enumerate(training_faces):
            _training_faces[index] = self.project_on_space(face)

        """
        Calculates the minimum allowed distance to a
        nearest face so that all the faces in the training_faces
        is classified as a face.
        """
        for training_face in _training_faces:
            minimal_distance_to_a_face = INT_MAX
            for face in self.face_space:
                minimal_distance_to_a_face =\
                    min(minimal_distance_to_a_face,
                        self.euclidian_distance(training_face, face))

            self.maximal_allowed_distance =\
                max(self.maximal_allowed_distance, minimal_distance_to_a_face)

        return self.maximal_allowed_distance

    def predict(self, image):
        """
        Predict on new face.

        The face is assumed to already be reshaped
        into the shape that were used for traning.
        """
        if self.face_space is None:
            print("Build face space before prediction")

            sys.exit(0)

        """Project the image onto the face space"""
        _i_in_facespace = self.project_on_space(image)

        min_distance_to_face = INT_MAX
        for face in self.face_space:
            min_distance_to_face =\
                min(min_distance_to_face,
                    self.euclidian_distance(face, _i_in_facespace))

        return (True, min_distance_to_face) if\
            min_distance_to_face <= self.maximal_allowed_distance else\
            (False, min_distance_to_face)


if __name__ == "__main__":

    time_stamp_pre_datafetch = time()

    (base_data, training_data, test_data) =\
        get_training_test_and_real_test_data()

    print("took {} seconds to read data"
          .format(time() - time_stamp_pre_datafetch))

    classifier = Eigenclassifier(base_dim=4, orthogonalize=True)
    classifier.build_base(base_data)
    classifier.get_min_distance_to_faces(base_data, training_data)
    classifier.show_principal_component()

