from typing import Callable
import tifffile
import glob
import os
import numpy as np
import cv2
import random
import torch

from DLIP.data.base_classes.base_dataset import BaseDataset


class BaseSegmentationDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        sample_filter,
        samples_dir: str = "samples",
        class_dir: str = "labels",
        samples_data_format="tif",
        labels_data_format="tif",
        transforms = None,
        empty_dataset=False,
        labels_available=True,
        return_trafos=False,
        label_transform: Callable = None,
        binarize_labels=False,
        return_weight_maps=False
    ):
        self.return_weight_maps = return_weight_maps
        self.empty_dataset = empty_dataset
        self.sample_filter = sample_filter
        self.labels_available = labels_available
        self.root_dir = root_dir
        self.samples_dir = samples_dir
        self.class_dir = class_dir
        self.samples_data_format = samples_data_format
        self.labels_data_format = labels_data_format
        self.return_trafos = return_trafos
        self.transforms = transforms
        self.label_transform = label_transform
        self.binarize_labels = binarize_labels
        self.internal_seed = 0

        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]


        self.samples = os.path.join(self.root_dir,self.samples_dir)
        if isinstance(self.class_dir,str):
            self.labels  = os.path.join(self.root_dir,self.class_dir)
        else:
            self.labels = [os.path.join(self.root_dir,x) for x in self.class_dir]

        # Get all sample names sorted as integer values
        all_samples_sorted = sorted(
            glob.glob(f"{self.samples}{os.path.sep}*"),
            key=lambda x: 
                x.split(f"{self.samples}{os.path.sep}")[1].split(
                    f".{samples_data_format}"
            ),
        )

        self.indices = []
        if not empty_dataset:
            # Extract indices from the sorted samples
            self.indices = [
                i.split(f"{self.samples}{os.path.sep}")[1].split(f".{samples_data_format}")[0]
                for i in all_samples_sorted
            ]
        self.raw_mode = False

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # load sample
        sample_path = os.path.join(self.samples, f"{self.indices[idx]}.{self.samples_data_format}")
        sample_img = tifffile.imread(sample_path) if self.samples_data_format=="tif" else cv2.imread(sample_path,-1)

        #weight_map_path = sample_path.replace(self.samples_dir,'weight_maps')
        weight_map_path = sample_path.replace(self.samples_dir,'weight_maps_filtered_20_percent')
        weight_map = tifffile.imread(weight_map_path) if self.samples_data_format=="tif" else cv2.imread(weight_map_path,-1)

        sample_img_lst = []
        weight_map_lst = []
        if isinstance(self.labels,str):
            label_lst = []
            trafo_lst = []
        elif isinstance(self.labels,list):
            label_lst = [[] for x in range(len(self.labels))]
            trafo_lst = [[] for x in range(len(self.labels))]
        else:
            raise ValueError('Cant load labels..')

        if self.labels_available:
            # load label map
            if isinstance(self.labels,str):
                label_path = os.path.join(self.labels, f"{self.indices[idx]}.{self.labels_data_format}")
                label_img = tifffile.imread(label_path) if self.labels_data_format=="tif" else cv2.imread(label_path,-1)
                label_one_hot = label_img
                if self.label_transform is not None:
                    label_one_hot = self.label_transform(label_one_hot)
            elif isinstance(self.labels,list):
                label_one_hot = list()
                for label_p in self.labels:
                    label_path = os.path.join(label_p, f"{self.indices[idx]}.{self.labels_data_format}")
                    label_img = tifffile.imread(label_path) if self.labels_data_format=="tif" else cv2.imread(label_path,-1)
                    lbl_one_hot = label_img
                    if self.label_transform is not None:
                        lbl_one_hot = self.label_transform(lbl_one_hot)
                    label_one_hot.append(lbl_one_hot)

        # raw mode -> no transforms
        if self.raw_mode:
            if self.labels_available:
                return sample_img,label_one_hot
            else:
                return sample_img
        
        if isinstance(label_one_hot,str):
            for transform in self.transforms:
                random.seed(self.internal_seed)
                im, lbl, trafo = transform(sample_img, label_one_hot)
                sample_img_lst.append(im)
                label_lst.append(lbl)
                trafo_lst.append(trafo)
                random.seed(self.internal_seed)
                _, lbl, _ = transform(sample_img, weight_map)
                weight_map_lst.append(lbl)
                self.internal_seed+=1
        elif isinstance(self.labels,list):
            for transform in self.transforms:
                random.seed(self.internal_seed)
                im, lbl, trafo = transform(sample_img, label_one_hot[0])
                sample_img_lst.append(im)
                label_lst[0].append(lbl)
                trafo_lst.append(trafo)
                random.seed(self.internal_seed)
                _, lbl, _ = transform(sample_img, weight_map)
                weight_map_lst.append(lbl)
                for i in range(1,len(label_one_hot)):
                    lbl_one_hot = label_one_hot[i]
                    random.seed(self.internal_seed)
                    im, lbl, trafo = transform(sample_img, lbl_one_hot)
                    label_lst[i].append(lbl)
                self.internal_seed+=1
            

        if len(sample_img_lst) == 1:
            sample_img_lst = sample_img_lst[0]
            weight_map_lst = weight_map_lst[0]
            if isinstance(label_one_hot,str):
                label_lst = label_lst[0].unsqueeze(2) if len(label_lst) > 0 else label_lst
            if isinstance(label_one_hot,list):
                for i in range(len(label_lst)):
                    label_lst[i] = label_lst[i][0].unsqueeze(2)
            trafo_lst = trafo_lst[0] if len(trafo_lst) > 0 else trafo_lst
       
        # sample_img_lst (optional: labels) (optional: trafos)
        if not self.return_trafos and not self.labels_available:
            return sample_img_lst
        if self.return_trafos and not self.labels_available:
            return sample_img_lst, trafo_lst
        if not self.return_trafos and self.labels_available and not self.return_weight_maps:
            return sample_img_lst, label_lst
        if not self.return_trafos and self.labels_available and self.return_weight_maps:
            return sample_img_lst, label_lst, weight_map_lst
        if self.return_trafos and self.labels_available:
            return sample_img_lst, label_lst, trafo_lst

    def pop_sample(self, index):
        return self.indices.pop(index)

    def add_sample(self, new_sample):
        self.indices.append(new_sample)

    def get_samples(self):
        return self.indices
