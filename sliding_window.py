from utilities.image import dim_reduction
from matplotlib.patches import Rectangle

class Window(object):

    def __init__(self, start, shape, data):
        self.start = start
        self.shape = shape
        self.data = data

        self.distance = None



    def get_classifiable(self, shape=(32, 32)):
        """Get classifiable dimension."""
        return dim_reduction(self.data, self.shape, shape)

    def set_distance(self, distance):
        """Set distance from positive."""
        self.distance = distance

        return self
    
    def add_rectangle_to_image(self, axis):
        """Add rectangle to a patch."""
        (width, heigth) = shape

        axis.add_patch(
            Rectangle(self.start, width, height, linewidth=1, edgecolor="r"))



def slide(image, shape, stride=10):
    """Slide through the image and return windows."""

    mx, my = image.shape
    x, y = shape

    sx, sy = 0, 0

    windows = []
    while sy + y < my:
        while sx + x < mx:
            from_x = sx
            to_x = sx + x

            from_y = sy
            to_y = sy + y

            data = image[from_x:to_x, from_y:to_y]

            windows.append(Window((sx, sy), shape, data))

            sx += stride

        sx = 0
        sy += stride

    return windows


def classify_windows(classifier, windows):
    positives, negatives = [], []
    for window in windows:
        (positive, distance) = classifier.predict(window.get_classifiable())

        if positive:
            positives.append(window.set_distance(distance))
        else:
            negatives.append(window.set_distance(distance))


    return positives, negatives

