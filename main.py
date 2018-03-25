from eigenfaces import Eigenclassifier
from utilities.data import get_data
from utilities.image import get_classifiable_image, vector_to_renderable, vector_to_image
from utilities.test import test
from sliding_window import Window, classify_windows, slide
from matplotlib.pyplot import imshow, subplots, show
from time import time

"""Train Classifier."""
pre_fetch = time()

(bdata, trdata, tedata) = get_data()

print("took {} seconds to read data"
      .format(time() - pre_fetch))

classifier = Eigenclassifier(base_dim=200)
classifier.build_base(bdata)
md, f, t = classifier.train(bdata, trdata)

success_frequency, _ = test(classifier, tedata)

print("Distance in face-space for faces {}".format(md))
print("Success frequency on faces {}".format(success_frequency))

"""Find Faces in Image."""
classifiable, original = get_classifiable_image("beatles.jpg")
fig, (ax1) = subplots(1, 1)
ax1.imshow(original)

pre_slide = time()
windows = slide(classifiable, (140, 140), stride=80)
print("took {} seconds to slide through all windows for size {}"
      .format(time() - pre_slide, 160))

pre_classification = time()
positives, negatives = classify_windows(classifier, windows)
print("took {} seconds to classify the windows"
      .format(time() - pre_classification))

for window in positives:
    window.add_rectangle_to_image(ax1)

show()



