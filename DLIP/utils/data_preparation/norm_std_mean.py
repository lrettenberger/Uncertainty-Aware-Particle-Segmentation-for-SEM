import numpy as np

# custom Albumentation transforms

def norm_std_mean(image, mean=None, std=None):
    image = image.astype("float32")

    if mean is None or std is None:
        mean = np.mean(image, axis=(0,1))
        std  = np.std(image, axis=(0,1))

    denominator = np.reciprocal(std+np.finfo(float).eps, dtype=np.float32)
    image = image.astype(np.float32)
    image -= mean
    image *= denominator
    return image
