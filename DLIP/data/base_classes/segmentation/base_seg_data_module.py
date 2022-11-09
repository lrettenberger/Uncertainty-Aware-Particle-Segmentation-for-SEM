import os
import random
import tifffile
import cv2
import numpy as np
import logging

from DLIP.data.base_classes.base_pl_datamodule import BasePLDataModule
from DLIP.data.base_classes.segmentation.base_seg_dataset import BaseSegmentationDataset

class GenericSegmentationDataModule(BasePLDataModule):
    def __init__(
        self,
        root_dir: str,
        n_classes: int,
        sample_filter,
        batch_size = 1,
        dataset_size = 1.0,
        val_to_train_ratio = 0,
        initial_labeled_ratio= 1.0,
        train_transforms=None,
        train_transforms_unlabeled=None,
        val_transforms=None,
        test_transforms=None,
        return_unlabeled_trafos=False,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
        class_dir='labels',
        binarize_labels = False,
    ):
        super().__init__(
            dataset_size=dataset_size,
            batch_size = batch_size,
            val_to_train_ratio = val_to_train_ratio,
            num_workers = num_workers,
            pin_memory = pin_memory,
            shuffle = shuffle,
            drop_last = drop_last,
            initial_labeled_ratio = initial_labeled_ratio,
        )
        self.root_dir = root_dir
        self.train_labeled_root_dir     = os.path.join(self.root_dir, "train")
        self.train_unlabeled_root_dir   = os.path.join(self.root_dir, "unlabeled")
        self.test_labeled_root_dir      = os.path.join(self.root_dir, "test")
        self.train_transforms = train_transforms
        self.train_transforms_unlabeled = (
            train_transforms_unlabeled
            if train_transforms_unlabeled is not None
            else train_transforms
        )
        self.sample_filter = sample_filter
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.return_unlabeled_trafos = return_unlabeled_trafos
        self.labeled_train_dataset: BaseSegmentationDataset = None
        self.unlabeled_train_dataset: BaseSegmentationDataset = None
        self.val_dataset: BaseSegmentationDataset = None
        self.test_dataset: BaseSegmentationDataset = None
        self.n_classes = n_classes
        self.samples_data_format, self.labels_data_format = 'tif','tif'
        self.class_dir = class_dir
        self.binarize_labels = binarize_labels
        self.__init_datasets()

    def __init_datasets(self):
        self.labeled_train_dataset = BaseSegmentationDataset(
            sample_filter = self.sample_filter,
            root_dir=self.train_labeled_root_dir, 
            transforms=self.train_transforms,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            class_dir = self.class_dir,
            binarize_labels = self.binarize_labels,
        )

        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
            self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))

        self.val_dataset = BaseSegmentationDataset(
            sample_filter = self.sample_filter,
            root_dir=self.train_labeled_root_dir, 
            transforms=self.val_transforms,
            empty_dataset=True,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            class_dir = self.class_dir,
            binarize_labels = self.binarize_labels,
        )

        self.unlabeled_train_dataset = BaseSegmentationDataset(
            sample_filter = self.sample_filter,
            root_dir=self.train_unlabeled_root_dir,
            transforms=self.train_transforms,
            labels_available=False,
            return_trafos=self.return_unlabeled_trafos,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            class_dir = self.class_dir,
            binarize_labels = self.binarize_labels,
        )
        
        self.test_dataset = BaseSegmentationDataset(
            sample_filter = self.sample_filter,
            root_dir=self.test_labeled_root_dir, 
            transforms=self.test_transforms,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            class_dir = self.class_dir,
            binarize_labels = self.binarize_labels,
        )

    def _determine_data_format(self):
        extensions = {"samples": list(), "labels": list()}

        for folder in extensions.keys():
            for file in os.listdir(os.path.join(self.train_labeled_root_dir,folder)):
                extensions[folder].append(os.path.splitext(file)[1].replace(".", ""))

            for file in os.listdir(os.path.join(self.train_unlabeled_root_dir,folder)):
                extensions[folder].append(os.path.splitext(file)[1].replace(".", ""))

        return max(set(extensions["samples"]), key = extensions["samples"].count),max(set(extensions["labels"]), key = extensions["labels"].count)


    def _determine_label_maps(self):
        map_lst = list()
        for file in os.listdir(os.path.join(self.train_labeled_root_dir,"labels")):
            file_path = os.path.join(self.train_labeled_root_dir,"labels", file)
            label_img = tifffile.imread(file_path) if self.labels_data_format=="tif" else cv2.imread(file_path,-1)
            map_lst.extend(np.unique(label_img))

        map_lst = sorted(map_lst)

        map_look_up = dict()
        if self.n_classes>1:
            for i in range(self.n_classes):
                map_look_up[i]= map_lst[i]
        else:
            map_look_up[0]= map_lst[-1]

        return map_look_up
    
    def init_val_dataset(self, split_lst=None):
        if len(self.labeled_train_dataset)>0:
            # default init is random
            if split_lst is None:
                num_val_samples = int(round(len(self.labeled_train_dataset) * (self.val_to_train_ratio)))
                num_val_samples = num_val_samples if num_val_samples > 0 else 1
                for _ in range(
                    num_val_samples
                ):
                    self.val_dataset.add_sample(
                        self.labeled_train_dataset.pop_sample(
                            random.randrange(len(self.labeled_train_dataset))
                        )
                    )

            else:
                ind_lst = []
                for elem in split_lst:
                    ind_lst.append(self.labeled_train_dataset.indices[elem])

                for ind_elem in ind_lst:
                    rem_ind = self.labeled_train_dataset.indices.index(ind_elem)
                    self.val_dataset.add_sample(
                        self.labeled_train_dataset.pop_sample(
                            rem_ind
                        )
                    )
        logging.warn("Init validation not possible due to no labeled training data.")
