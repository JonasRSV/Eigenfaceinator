from eigenfaces import Eigenclassifier
from utilities.data import get_data
from utilities.image import get_classifiable_image, vector_to_renderable, vector_to_image
from sliding_window import Window, classify_windows, slide
from matplotlib.pyplot import imshow, subplots, show
from time import time
import sys

"""Train Classifier."""
pre_fetch = time()

(base_data, training_data, test_data) = get_data()

print("took {} seconds to read data"
      .format(time() - pre_fetch))

classifier = Eigenclassifier(base_dim=40, orthogonalize=True)
classifier.build_base(base_data)
md, f, t = classifier.set_allowed_distance(base_data, training_data)

fig, (ax1, ax2) = subplots(1, 2)

print(md)
ax1.imshow(vector_to_image(f, (32, 32)))
ax2.imshow(vector_to_image(t, (32, 32)))
show()

"""Find Faces in Image."""
classifiable, original = get_classifiable_image("avengers.jpg")
fig, (ax1, ax2, ax3) = subplots(1, 3)



for dim in range(32, 60):
    windows = slide(classifiable, (dim, dim), stride=50)

    ax1.imshow(vector_to_image(windows[0].data, windows[0].shape))
    ax2.imshow(vector_to_image(windows[0].get_classifiable(), (32, 32)))
    p, d, face = classifier.predict(windows[0].get_classifiable())

    ax3.imshow(vector_to_image(face, (32, 32)))

    print(p)
    print(d)


    break



    print("Slided through windows once")
    positives, negatives = classify_windows(classifier, windows)

    print("Nof positives {}, nOf negatives {}".format(len(positives), len(negatives)))


show()



