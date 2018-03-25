from pandas import read_csv
from random import choice
from numpy import array, zeros, fromstring
from utilities.image import dim_reduction


def get_data():
    """Read and reformat traning data."""
    data = read_csv("training.csv", sep=',', header='infer')
    test = read_csv("test.csv", sep=",", header="infer")

    images = zeros((7049, 9216))
    test_images = zeros((1783, 9216))
    for index, img in enumerate(data["Image"]):
        images[index] = fromstring(img, dtype=int, sep=" ")

    for index, img in enumerate(test["Image"]):
        test_images[index] = fromstring(img, dtype=int, sep=" ")

    return (choose_images(images, cardinality=2000),
            choose_images(images, cardinality=3000),
            choose_images(test_images, cardinality=500))


def choose_images(images, cardinality=500):
    """
    Choose a random number of images.

    This is currently flawed because it might
    choose duplicates at the moment.
    """
    training_images = zeros((cardinality, 32 * 32))

    for index in range(cardinality):
        training_images[index] =\
            dim_reduction(choice(images))

    return training_images

