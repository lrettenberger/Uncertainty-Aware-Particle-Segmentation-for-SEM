import random
import numpy as np

from DLIP.data.base_classes.segmentation.base_seg_data_module import GenericSegmentationDataModule
from DLIP.data.soft_label_synth.soft_label_dataset import SoftLabelDataset


class SoftLabelDataModule(GenericSegmentationDataModule):
    
    
    def __init__(
        self,
        root_dir: str,
        n_classes: int,
        batch_size=1,
        dataset_size=1,
        val_to_train_ratio=0.2,
        initial_labeled_ratio=1,
        train_transforms=None,
        train_transforms_unlabeled=None,
        val_transforms=None,
        test_transforms=None,
        return_unlabeled_trafos=False,
        num_workers=42,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
        hard_labeling=True
    ):
        if hard_labeling:
            self.class_dir = 'labels_hard'
            self.label_transform = lambda x: np.round(np.expand_dims(x/255., 2))
        else:
            self.class_dir = 'labels_soft'
            self.label_transform = lambda x: np.concatenate((np.expand_dims(x/255,2),np.expand_dims(1 - x/255,2)),axis=2)
        super().__init__(
            root_dir,
            n_classes,
            batch_size=batch_size,
            dataset_size=dataset_size,
            val_to_train_ratio=val_to_train_ratio,
            initial_labeled_ratio=initial_labeled_ratio,
            train_transforms=train_transforms,
            train_transforms_unlabeled=train_transforms_unlabeled,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            return_unlabeled_trafos=return_unlabeled_trafos,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last
        )
        self.__init_datasets()
        self.labeled_train_dataset = self.train_dataset
        
    def __init_datasets(self):
        self.train_dataset = SoftLabelDataset(
            root_dir=self.train_labeled_root_dir, 
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            class_dir=self.class_dir,
            label_transform=self.label_transform,
            transforms=self.train_transforms
        )
        
        self.val_dataset = SoftLabelDataset(
            root_dir=self.train_labeled_root_dir, 
            transforms=self.val_transforms,
            empty_dataset=True,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            class_dir=self.class_dir,
            label_transform=self.label_transform,
        )

        for _ in range(int(len(self.train_dataset) * (self.val_to_train_ratio))):
            popped = self.train_dataset.pop_sample(random.randrange(len(self.train_dataset)))
            self.val_dataset.add_sample(popped)
       
        self.test_dataset = SoftLabelDataset(
            root_dir=self.test_labeled_root_dir, 
            transforms=self.test_transforms,
            samples_data_format=self.samples_data_format,
            labels_data_format=self.labels_data_format,
            class_dir=self.class_dir,
            label_transform=self.label_transform,
        )
