import numpy as np

def get_particle_sizes(inst_segmentations):
    sizes_by_pixels = [np.sum(inst_segmentations == x) for x in np.unique(inst_segmentations) if x > 0]
    return sizes_by_pixels