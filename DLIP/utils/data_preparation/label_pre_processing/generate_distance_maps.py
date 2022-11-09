import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import label2rgb
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt, grey_closing, generate_binary_structure
from skimage import measure
from skimage.morphology import disk
import tifffile


class DistanceMapGenerator:
    def __init__(
            self, 
            root_dir,
            label_dir = "labels") -> None:
        root_dir = root_dir
        self.label_dir = label_dir
        self.data_dirs = [os.path.join(root_dir,"train"), os.path.join(root_dir,"test")]

    def generate_labels(self, folder_name):
        for data_dir in self.data_dirs:
            if not os.path.exists(os.path.join(data_dir, folder_name)):
                os.makedirs(os.path.join(data_dir, folder_name))
            for file in os.listdir(os.path.join(data_dir,self.label_dir)):
                file_path_src = os.path.join(data_dir, self.label_dir, file)
                file_path_dst = os.path.join(data_dir, folder_name, file)

                print(file)

                label_raw = cv2.imread(file_path_src,-1)
                print(label_raw.shape)

                label_dist = self._dist_trafo( label_raw)
                #label_rgb = label2rgb(label_raw[:,:,0],bg_label=0)

                tifffile.imwrite(file_path_dst.replace("png","tif"),label_dist.astype(np.float32))
                # plt.imshow(label_dist)
                # plt.show()

    def _dist_trafo(self, label):
         # Preallocation
        label_dist = np.zeros(shape=label.shape, dtype=np.float)

        # Find centroids, calculate distance transforms
        props = measure.regionprops(label)
        for i in range(len(props)):
            # Get nucleus and Euclidean distance transform for each nucleus
            nucleus = (label == props[i].label)
            centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
            nucleus_crop_dist = distance_transform_edt(nucleus)
            if np.max(nucleus_crop_dist) > 0:
                nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
            label_dist += nucleus_crop_dist

        return label_dist


if __name__ == "__main__":
    root_dir = "/home/ws/kg2371/datasets/sem_segmentation_sep_2022"

    dmg_obj = DistanceMapGenerator(root_dir)
    dmg_obj.generate_labels("labels_dist_map")
