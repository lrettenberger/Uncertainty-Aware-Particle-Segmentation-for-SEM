import matplotlib
matplotlib.use('Agg')
from skimage.segmentation import find_boundaries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_holes
from glob import glob
from tqdm import tqdm
import tifffile
from pathlib import Path


def make_weight_map(masks):
    masks = (masks > 0).astype(int)
    weight_map = np.zeros_like(masks)
    for i, mask in enumerate(masks):
        dist_transform = ndimage.distance_transform_edt(mask) 
        dist_transform *= 255.0/dist_transform.max()
        weight_map[i] = dist_transform
    return np.sum(weight_map,axis=0)


for phase in ['train','val','test']:
    masks = glob(f'/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/{phase}/labels/*.tif')
    Path(f'/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/{phase}/dist_maps/').mkdir(parents=True, exist_ok=True)


    for mask_path in tqdm(masks):
        
        for mask_path in tqdm(masks):
            mask = tifffile.imread(mask_path)
        mask_split = np.array([(mask==i)*1 for i in range(1,int(np.max(mask)+1))])
        mask_split_filled = np.zeros_like(mask_split)
        for i in range(len(mask_split)):
            others_mask = np.sum(np.concatenate((mask_split[0:i],mask_split[i+1:]),axis=0),axis=0)==0
            filled_mask = binary_fill_holes(mask_split[i])
            filled_filtered = np.logical_and(filled_mask,others_mask)
            mask_split_filled[i] = filled_filtered
        mask_recombined = np.zeros_like(mask)
        for i in range(len(mask_split_filtered)):
            mask_recombined = mask_recombined + mask_split_filtered[i]*(i+1)
        mask_recombined = mask_recombined.astype(np.uint8)
        success = cv2.imwrite(mask_path.replace('labels',f'labels_{REMOVED_PARTICLES_SUFFIX}'),mask_recombined)
        if not success:
            raise Exception(f"Could not save sample {mask_path.replace('labels',f'labels_{REMOVED_PARTICLES_SUFFIX}')}")
        
        mask = tifffile.imread(mask_path)
        mask_split = np.array([(mask==i)*1 for i in range(1,int(np.max(mask)+1))])
        mask_split = np.array([binary_fill_holes(x) for x in mask_split])
        if np.sum(mask_split) > 0 :
            weights = make_weight_map(mask_split)
        else:
            weights = np.zeros_like(mask)
        cv2.imwrite(mask_path.replace('labels','dist_maps'),(weights/255.0).astype(np.float32))