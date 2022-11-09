"""
    Example implementation of a datamodule. Does not furfill any purpose apart from providing guidance.
"""

import random
from DLIP.data.base_classes.base_pl_datamodule import BasePLDataModule
from DLIP.data.example.example_dataset import ExampleDataset


class ExamplePLDatamodule(BasePLDataModule):
    
    def __init__(
        self,
        number_of_images: int,
        dataset_size: float = 0.8,
        batch_size: int = 32,
        val_to_train_ratio: int = 0.2,
        initial_labeled_ratio: float = 0.5,
        num_workers: int = 5,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = False,
        train_transforms = None,
        val_transforms = None,
        test_transforms = None
    ):
        super().__init__(
            dataset_size,
            batch_size,
            val_to_train_ratio,
            initial_labeled_ratio,
            num_workers,
            pin_memory,
            shuffle,
            drop_last
        )
        self.labeled_train_dataset = ExampleDataset(num_of_images=number_of_images, labels_available=True)
        self.unlabeled_train_dataset = ExampleDataset(num_of_images=0, labels_available=False)
        self.val_dataset = ExampleDataset(num_of_images=0, labels_available=True)
        self.test_dataset = ExampleDataset(num_of_images=5, labels_available=False)

        # > len(self.labeled_train_dataset) = 100

        # Since the dataset_size is set to 0.8 drop 20% of the samples from the labeled train dataset
        # to only maike 80% of the samples available for furhter processing.
        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
            self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))
        
        # > len(self.labeled_train_dataset) = 80
        
        # Since the initial_labeled_size is set to 0.5, remove half of the samples form the labeled dataset
        # and add it to the unalebeld dataset.
        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.initial_labeled_ratio))):
            popped = self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))
            self.unlabeled_train_dataset.add_sample(popped[0])
            
        # > len(self.labeled_train_dataset) = 40
        # > len(self.unlabeled_train_dataset) = 40
        
        # Since 20% should be used for validation take those 20% from the labeled train set and add them
        # to the val dataset
        for _ in range(int(len(self.labeled_train_dataset) * (self.val_to_train_ratio))):
            popped = self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))
            self.val_dataset.add_sample(popped)
            
        # > len(self.labeled_train_dataset) = 32
        # > len(self.val_dataset) = 8
