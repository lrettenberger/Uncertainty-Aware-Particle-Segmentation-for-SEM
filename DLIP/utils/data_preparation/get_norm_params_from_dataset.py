import logging
import numpy as np
from tqdm import tqdm

def get_norm_params_from_dataset(dataset):
    '''
    Compute the mean and std value of dataset.
    '''
    dummy_img, _ = dataset[0]
    dim = dummy_img.ndim

    mean    = np.zeros(3) if dim==3 else np.zeros(1)
    std     = np.zeros(3) if dim==3 else np.zeros(1)
    logging.info("Computing mean and std...")
    for img, label in tqdm(dataset):
        mean += np.mean(img, axis=(0, 1))
        std  += np.std(img, axis=(0, 1))
    logging.info("...done")
    mean /= len(dataset)
    std  /= len(dataset)

    return mean, std
