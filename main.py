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

classifier = Eigenclassifier(base_dim=200)
classifier.build_base(base_data)
md, f, t = classifier.set_allowed_distance(base_data, training_data)

# fig, (ax1, ax2) = subplots(1, 2)
# ax1.imshow(vector_to_image(f, (32, 32)))
# ax2.imshow(vector_to_image(t, (32, 32)))
# show()

"""Find Faces in Image."""
classifiable, original = get_classifiable_image("cbeatles.jpg")
fig, (ax1) = subplots(1, 1)

ax1.imshow(original)

print("Distance in face-space for faces {}".format(md))
pre_slide = time()
windows = slide(classifiable, (50, 50), stride=20)
print("took {} seconds to slide through all windows for size {}"
      .format(time() - pre_slide, 50))

pre_classification = time()
positives, negatives = classify_windows(classifier, windows)
print("took {} seconds to classify the windows"
      .format(time() - pre_classification))

for window in positives:
    window.add_rectangle_to_image(ax1)

show()



