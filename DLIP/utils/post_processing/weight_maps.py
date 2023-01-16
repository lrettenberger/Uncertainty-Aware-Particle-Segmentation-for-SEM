from glob import glob
from sklearn.utils import shuffle
import tifffile
import cv2
import numpy as np
from patchify import patchify
from skimage.segmentation import watershed
from skimage import measure
from tqdm import tqdm
from pathlib import Path

from skimage.segmentation import find_boundaries

w0 = 10
sigma = 5

def make_weight_map(masks):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.
    
    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents
	one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)
    
    """
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ


# weight maps
# val durchgucken -> einmal leer?


for phase in ['train','val','test']:

    masks = glob(f'/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/{phase}/labels_filtered_20_percent/*.tif')
    
    # creating dirs
    Path(f'/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/{phase}/weight_maps_filtered_20_percent/').mkdir(parents=True, exist_ok=True)

    for mask_path in tqdm(masks):
        mask = tifffile.imread(mask_path)
        mask = cv2.resize(mask,(600,600),interpolation=cv2.INTER_NEAREST)
        mask_split = np.array([(mask==i)*1 for i in range(1,int(np.max(mask)+1))])
        mask_split = np.array([x for x in mask_split if np.max(x) > 0])
        if len(mask_split) == 0:
            tifffile.imwrite(mask_path.replace('labels','weight_maps'),np.ones((1200,1200)))
            continue
        weight = make_weight_map(mask_split)
        # scaled to be in [1,2]
        weight = (2-1)*((weight-np.min(weight))/(np.max(weight)-np.min(weight))+1)
        weight = cv2.resize(weight,(1200,1200),interpolation=cv2.INTER_NEAREST)
        tifffile.imwrite(mask_path.replace('labels','weight_maps'),weight)
