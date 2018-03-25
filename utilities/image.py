from numpy import full, array, zeros, sum, asarray
from PIL import Image
import sys


def dim_reduction(vector, current=(96, 96), target=(32, 32)):
    """Reformat a numpy vector of image to new format vector."""
    matrix = vector.reshape(current)
    (cx, cy) = current
    (tx, ty) = target

    x_filter = int(cx / tx)
    y_filter = int(cy / ty)

    if x_filter == 0 or y_filter == 0:
        print("Cannot increase dimension, please let window be atleast 32 x 32")
        sys.exit(0)


    rfilter = full((x_filter, y_filter), 1 / (x_filter * y_filter))

    reduced = zeros(target)
    for x in range(tx):
        for y in range(ty):
            x_from = x * x_filter
            x_to = x_from + x_filter

            y_from = y * y_filter
            y_to = y_from + y_filter

            reduced[x, y] =\
                sum(rfilter * matrix[x_from:x_to, y_from:y_to])

    return reduced.flatten()


def vector_to_image(vector, shape):
    """Imagize a vector."""
    im_dim = vector.reshape(shape)
    image = zeros((shape[0], shape[1], 3))

    kind_of_abs = lambda x: x if x > 0 else 255
    for xi, row in enumerate(im_dim):
        for yi, cell in enumerate(row):
            image[xi, yi] = [kind_of_abs(im_dim[xi, yi]) / 256] * 3

    return image


def normalize_vector_image(vector):
    """Shapes vector into displayable format."""
    mn = min(vector)
    mx = max(vector)

    stabiliser = mx + abs(mn)

    avoid_zero_division = stabiliser if stabiliser != 0 else 1

    vector = vector + abs(mn)
    return vector * (256 / avoid_zero_division)


def vector_to_renderable(vector, shape):
    """Make vector plottable."""
    normalized = normalize_vector_image(vector)
    return vector_to_image(normalized, shape)


def grey_scale(image,
               wpixel=lambda x: x[0] * 0.25 + x[1] * 0.5 + x[2] * 0.25):
    """Reduce image channels from 3 to 1."""
    if len(image.shape) == 2:
        return image
    else:
        (x, y, _) = image.shape
        grey = zeros((x, y))
        for col in range(x):
            for row in range(y):
                grey[col][row] = wpixel(image[col][row])

        return grey


def load_image(name):
    """Load image."""
    img = Image.open(name)
    img.load()

    return asarray(img, dtype="B")


def get_classifiable_image(name):
    """Get image to do sliding window on."""
    img = load_image(name)
    classifiable = grey_scale(img)
    return classifiable, img

