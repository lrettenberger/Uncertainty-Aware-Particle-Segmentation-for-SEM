import numpy as np
import tifffile
from glob import glob

from pathlib import Path

inMin = 1
inMax = 1200*1200
outMin = 0.01
outMax = 1

def mapRange(value):
    val = outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))
    return 1-val


for phase in['train','val','test']: 
    labels = glob(f'/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/{phase}/labels_filtered_20_percent/*')

    save_dir = '/'.join(map(str, labels[0].replace('labels_filtered_20_percent','weight_maps_weighted_20_percent').split('/')[:-1]))
    Path(save_dir).mkdir(parents=True, exist_ok=True)


    for i in range(len(labels)):
        mask = tifffile.imread(labels[i])
        mask_split = np.array([(mask==j)*1 for j in np.unique(mask)])
        weight_values = [mapRange(np.sum(mask_split[j])) for j in range(len(mask_split))]

        weight_map = np.zeros_like(mask)

        for j in range(1,len(weight_values)):
            weight_map = weight_map + (weight_values[j]*(mask==j))
            
        unweighted_weight_map = tifffile.imread(labels[i].replace('labels_filtered_20_percent','weight_maps_filtered_20_percent'))

        weight_map = weight_map*weight_map
        weight_map[weight_map==0] = 1

        comb = weight_map * unweighted_weight_map

        tifffile.imwrite(labels[i].replace('labels_filtered_20_percent','weight_maps_weighted_20_percent'),comb)