from utilities.data import get_data
from eigenfaces import Eigenclassifier



def test(classifier, data):

    a_failure = None
    closest_to_failure = None

    positives = 0
    negatives = 0
    for face in data:
        positive, distance, closest = classifier.predict(face)

        if positive:
            positives += 1
        else:
            negatives += 1

            a_failure = face
            closest_to_failure = closest



    success_rate = positives / (positives + negatives)

    return success_rate, (a_failure, closest_to_failure)

