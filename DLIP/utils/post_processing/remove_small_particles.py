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



# masks = glob('/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/train/labels/*.tif')
# particle_sizes = []
# for mask_path in tqdm(masks):
#     mask = tifffile.imread(mask_path)
#     mask_split = np.array([(mask==i)*1 for i in range(1,int(np.max(mask)+1))])
#     mask_split = np.array([binary_fill_holes(x) for x in mask_split])
#     particle_sizes =  particle_sizes + [np.sum(x) for x in mask_split]
# particle_sizes = np.sort(particle_sizes)
# for i in [0.1,0.15,0.2,0.3,0.4,0.5]:
#     print(f'Lower {int(i*100)}% : {particle_sizes[int(len(particle_sizes)*i)]}')
# Lower 10% : 713
# Lower 15% : 985
# Lower 20% : 1346
# Lower 30% : 2347
# Lower 40% : 3918
# Lower 50% : 6142

# 656 contains the 20% smallest particles
SIZE_THRESHOLD = 1346
REMOVED_PARTICLES_SUFFIX = '_filtered_20_percent'

# for normal labels
SIZE_THRESHOLD = -1
REMOVED_PARTICLES_SUFFIX = ''

print('Creating filtered labels and distance maps')

for phase in ['train','val','test']:
    print(f'Removing small particles for {phase} samples.')

    masks = glob(f'/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/{phase}/labels/*.tif')
    
    # creating dirs
    Path(f'/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/{phase}/labels{REMOVED_PARTICLES_SUFFIX}/').mkdir(parents=True, exist_ok=True)
    Path(f'/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/{phase}/dist_maps{REMOVED_PARTICLES_SUFFIX}/').mkdir(parents=True, exist_ok=True)

    for mask_path in tqdm(masks):
        mask = tifffile.imread(mask_path)
        mask_split = np.array([(mask==i)*1 for i in range(1,int(np.max(mask)+1))])
        mask_split_filled = np.zeros_like(mask_split)
        for i in range(len(mask_split)):
            others_mask = np.sum(np.concatenate((mask_split[0:i],mask_split[i+1:]),axis=0),axis=0)==0
            filled_mask = binary_fill_holes(mask_split[i])
            filled_filtered = np.logical_and(filled_mask,others_mask)
            mask_split_filled[i] = filled_filtered
        mask_split_filtered = [x for x in mask_split_filled if np.sum(x)>SIZE_THRESHOLD]
        mask_recombined = np.zeros_like(mask)
        for i in range(len(mask_split_filtered)):
            mask_recombined = mask_recombined + mask_split_filtered[i]*(i+1)
        mask_recombined = mask_recombined.astype(np.uint8)
        success = cv2.imwrite(mask_path.replace('labels',f'labels{REMOVED_PARTICLES_SUFFIX}'),mask_recombined)
        if not success:
            raise Exception(f"Could not save sample {mask_path.replace('labels',f'labels{REMOVED_PARTICLES_SUFFIX}')}")
        
        # we also need new distance maps
        
        mask = mask_recombined
        mask_split = np.array([(mask==i)*1 for i in range(1,int(np.max(mask)+1))])
        mask_split = np.array([binary_fill_holes(x) for x in mask_split])
        for i in range(len(mask_split)):
            other_masks = np.sum(mask_split)
        if np.sum(mask_split) > 0 :
            weights = make_weight_map(mask_split)
        else:
            weights = np.zeros_like(mask)
        success = cv2.imwrite(mask_path.replace('labels',f'dist_maps{REMOVED_PARTICLES_SUFFIX}'),(weights/255.0).astype(np.float32))
        if not success:
            raise Exception(f"Could not save sample {mask_path.replace('labels',f'dist_maps{REMOVED_PARTICLES_SUFFIX}')}")