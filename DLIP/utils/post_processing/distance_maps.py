from skimage.segmentation import find_boundaries
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    #masks = masks[0,:].detach().cpu().numpy()
    nrows, ncols = 600,600
    # masks_binary = np.zeros((np.max(masks)-1,512,512))
    # for i in range(1,np.max(masks)):
    #     masks_binary[i-1] = (masks == i)*1
    # masks = masks_binary
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
        if len(xSum) == 0 and len(ySum) == 0:
            distMap[:, i] = np.ones_like(distMap[:,i])
        else:
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


import tifffile
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label
from skimage.segmentation import watershed
from skimage import measure
from glob import glob
from tqdm import tqdm

imgs = glob('/home/ws/kg2371/datasets/sem_segmentation_sep_2022/test/labels/*.tif')

for img_path in tqdm(imgs):
    img = tifffile.imread(img_path)
    global_max_instance = 0
    masks_binary = np.zeros(img.shape)
    for i in range(1,np.max(img)+1):
        seeds   = measure.label((img==i)*1, background=0)
        masks   = (img==i)*1
        gt_mask = watershed(image=-((img==i)*1), markers=seeds, mask=masks, watershed_line=False)
        gt_mask += global_max_instance
        gt_mask[gt_mask==global_max_instance] = 0
        masks_binary += gt_mask
        global_max_instance = int(np.max(masks_binary))
    tifffile.imwrite(img_path.replace('labels','labels_clean'),masks_binary.astype(np.uint8))
print()   
    

# params = [(20, 16, 10), (44, 16, 10), (47, 47, 10)]
# masks = np.zeros((3, 64, 64))
# for i, (cx, cy, radius) in enumerate(params):
#     cv2.circle(masks[i],(cx,cy),radius,color=(255, 0, 0),thickness=-1)



# imgs = glob('/home/ws/kg2371/datasets/sem_segmentation_sep_2022/train/labels_clean/*.tif')

# for img in tqdm(imgs):
#     masks = tifffile.imread(img).astype(np.uint8)
#     if np.max(masks) == 0:
#         tifffile.imwrite(img.replace('labels_clean','weight_maps'),np.ones((600,600)))
#         continue

#     masks_binary = np.zeros((np.max(masks),600,600))
#     for i in range(1,np.max(masks)+1):
#         masks_binary[i-1] = (masks==i)*255
#     masks = masks_binary  
    

        

   # weights = make_weight_map(masks)
   # tifffile.imwrite(img.replace('labels_clean','weight_maps'),weights*(1/np.max(weights)))