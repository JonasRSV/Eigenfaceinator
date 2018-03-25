import sys
from utilities import data
from utilities import image
from numpy import array, mean, zeros, sum, vectorize, argsort, sqrt
from numpy.linalg import eig, norm
from matplotlib import pyplot as plt
from time import time

"""True intmax does not exists in python3 but this'll do."""
INT_MAX = 100000000000000


class Eigenclassifier(object):
    """Classifier scope."""

    def __init__(self, base_dim=20, verbose=True, orthogonalize=False):
        """Constructor."""
        self.base_dim = base_dim
        self.eigen_base = None
        self.mean_face = None
        self.covariance = None
        self.expected_values = None
        self.verbose = verbose
        self.orthogonalize = orthogonalize

        """Used in classification."""
        self.face_space = None
        self.maximal_allowed_distance = 0

        """Used to print pretty images."""
        self.faces = None
        self.training_faces = None

    def project_on_space(self, vector):
        """Project the vector onto the space."""
        base_projection = zeros(self.base_dim)
        for index, base_vector in enumerate(self.eigen_base):
            axis = zeros(self.base_dim)
            axis[index] = vector @ base_vector

            base_projection = base_projection + axis

        return base_projection

    def euclidian_distance(self, v1, v2):
        """Calculate the euclidian distance of two vectors."""
        return sqrt((v1 - v2) @ (v1 - v2))

    def show_principal_component(self, shape=(32, 32)):
        """Show the eigen face of the principal component."""
        if self.eigen_base is None:
            sys.stderr.write(
                "Must build base before trying to show the classifier")
            sys.exit(1)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

        principal_component_0 =\
            image.normalize_vector_image(self.covariance @ self.eigen_base[0])
        principal_component_1 =\
            image.normalize_vector_image(self.covariance @ self.eigen_base[1])
        principal_component_2 =\
            image.normalize_vector_image(self.covariance @ self.eigen_base[2])
        principal_component_3 =\
            image.normalize_vector_image(self.covariance @ self.eigen_base[3])

        ax1.imshow(image.vector_to_image(principal_component_0, shape))
        ax2.imshow(image.vector_to_image(principal_component_1, shape))
        ax3.imshow(image.vector_to_image(principal_component_2, shape))
        ax4.imshow(image.vector_to_image(principal_component_3, shape))

        plt.tight_layout()
        plt.show()

    def build_base(self, training_data):
        """Build the eigen base."""
        time_stamp_prebuild = time()
        self.mean_face = mean(training_data, axis=0)

        """ This broadcasts the mean_vector through the matrix """
        self.expected_values = training_data - self.mean_face

        (_, NN) = self.expected_values.shape

        self.covariance = self.expected_values.T @ self.expected_values

        (scalars, vectors) = eig(self.covariance)

        if len(scalars) < self.base_dim:
            print("Cannot build eigen base of " + self.base_dim
                  + " size, not enough eigen values exist "
                  + str(len(scalars)) + " eigen values were found")

            sys.exit(1)

        """stripps the vectors of the imaginary part"""
        im_stripper = vectorize(lambda x: x.real)

        """
        Gets the index of the #base_dim number of largest
        eigen scalars
        """
        max_eig_index = argsort(scalars)[-self.base_dim:]
        self.eigen_base = zeros((self.base_dim, NN))
        for index, eigen in enumerate(max_eig_index):
            scalar = scalars[eigen]
            vector = vectors[eigen]

            if abs(scalar.imag) > 0.00001:
                print("To big imag parts")
                sys.exit(1)

            im_stripped = im_stripper(vector)
            self.eigen_base[index] = im_stripped / norm(im_stripped)

        if self.orthogonalize:
            for i in range(self.base_dim):
                for j in range(i + 1, self.base_dim):
                    self.eigen_base[i] =\
                        self.eigen_base[i] -\
                        self.eigen_base[j] @ self.eigen_base[i]\
                        * self.eigen_base[j]

                    self.eigen_base[i] = self.eigen_base[i]\
                        / norm(self.eigen_base[i])

        if self.verbose:
            print(
                "base took {} seconds to build".format(
                    time() - time_stamp_prebuild
                ))

        return self.eigen_base

    def train(self, faces, training_faces, calc_variance=True):
        """Find minimal distance for each training face to a face."""
        if self.eigen_base is None:
            print("Build base before space dummy")

            sys.exit(0)

        pre_train = time()
        """Reset maximal_allowed distance."""
        self.maximal_allowed_distance = 0
        self.faces = faces
        self.training_faces = training_faces

        faces = faces - self.mean_face
        training_faces = training_faces - self.mean_face

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

        total_distance = 0
        most_distant_faces = 0
        distances = []
        mdf1 = None
        mdf2 = None
        for it, training_face in enumerate(_training_faces):
            minimal_distance_to_a_face = INT_MAX

            closest_face = INT_MAX
            cf = None

            for ir, face in enumerate(self.face_space):
                minimal_distance_to_a_face =\
                    min(minimal_distance_to_a_face,
                        self.euclidian_distance(training_face, face))

                if closest_face > minimal_distance_to_a_face:
                    closest_face = minimal_distance_to_a_face
                    cf = self.faces[ir]

            distances.append(minimal_distance_to_a_face)
            total_distance += minimal_distance_to_a_face
            self.maximal_allowed_distance =\
                max(self.maximal_allowed_distance, minimal_distance_to_a_face)

            if most_distant_faces < self.maximal_allowed_distance:
                most_distant_faces = self.maximal_allowed_distance
                mdf1 = self.training_faces[it]
                mdf2 = cf

        mean_distance = total_distance / len(self.training_faces)
        print("Max distance {}".format(self.maximal_allowed_distance))
        print("Mean Distance {}".format(mean_distance))
        print("time to train {}".format(time() - pre_train))

        return mean_distance, mdf1, mdf2

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
        image = image - self.mean_face
        _i_in_facespace = self.project_on_space(image)

        closest_face = None
        previous_md = INT_MAX
        min_distance_to_face = INT_MAX
        for index, face in enumerate(self.face_space):
            min_distance_to_face =\
                min(min_distance_to_face,
                    self.euclidian_distance(face, _i_in_facespace))

            if previous_md > min_distance_to_face:
                previous_md = min_distance_to_face
                closest_face = self.faces[index]

        return (True, min_distance_to_face, closest_face) if\
            min_distance_to_face <= self.maximal_allowed_distance else\
            (False, min_distance_to_face, closest_face)


if __name__ == "__main__":

    pre_fetch = time()

    (bdata, trdata, tedata) = data.get_data()

    print("took {} seconds to read data"
          .format(time() - pre_fetch))

    classifier = Eigenclassifier(base_dim=5, orthogonalize=True)
    classifier.build_base(bdata)
    classifier.train(bdata, trdata)
    classifier.show_principal_component()

