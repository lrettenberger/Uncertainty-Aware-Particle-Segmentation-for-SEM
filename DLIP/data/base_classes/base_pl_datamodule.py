"""
    The base PyTorch Lightning (PL) DataModule. A new PL-Datamodule should extend this base class, 
    to ensure full flexibility and usability. This datamodule provides functionality to habe a partially
    labeled datamoudle, which is reflected in the loader functions.
"""
import logging
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from DLIP.data.base_classes.base_dataset import BaseDataset

from DLIP.utils.data_preparation.custom_collate import custom_collate
from DLIP.utils.data_preparation.seed_worker import seed_worker


class BasePLDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            dataset_size: float,
            batch_size: int,
            val_to_train_ratio: int,
            initial_labeled_ratio: float,
            num_workers: int,
            pin_memory: bool,
            shuffle: bool,
            drop_last: bool,
        ):
        """ Init the data module.

        Args:
            dataset_size (float): 
                The overall dataset size. With this parameter the relative size of the dataset can be reduced.
                1.0 means full size. 0.5 means that only half of the samples are available.
            batch_size (int):
                Batch size. Used by PL-Lightning.
            val_to_train_ratio (int):
                How many samples should be used vor validation. Relative value. 
                0.2 means 20\% of samples are used for validation.
            initial_labeled_ratio (float):
                How many samples should be initially labeled.
            num_workers (int):
                Number of workers (processes) used by PL-Lightning.
            pin_memory (bool):
                Whether memory should be pinned by PL-Lightning.
            shuffle (bool):
                If samples should be shuffeled.
            drop_last (bool):
                Wheteher the last batch should be dropped if it is smaller than the batch size.
                This happens if the total number of samples is not divisble by the batch size.
        """
        super().__init__()
        self.dataset_size       = dataset_size
        self.batch_size         = batch_size
        self.val_to_train_ratio = val_to_train_ratio
        self.num_workers        = num_workers
        self.pin_memory         = pin_memory
        self.shuffle            = shuffle
        self.drop_last          = drop_last
        self.initial_labeled_ratio = initial_labeled_ratio
        # The following variables should be initialized by the implementing class
        self.labeled_train_dataset: BaseDataset = None
        self.unlabeled_train_dataset: BaseDataset = None
        self.val_dataset: BaseDataset = None
        self.test_dataset: BaseDataset = None
        logging.info(f"Using {self.num_workers} workers for data loading")
        

    def train_dataloader(self, get_labeled_share=True):
        if get_labeled_share:
            return self._labeled_train_dataloader()
        return self._unlabeled_train_dataloader()


    def _labeled_train_dataloader(self):
        if self.labeled_train_dataset is None:
            raise NameError('labeled_train_dataset has not been initialized!')
        logging.info(f"Getting Labeled Train Dataset with length {len(self.labeled_train_dataset)}")
        return DataLoader(
            self.labeled_train_dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )


    def _unlabeled_train_dataloader(self):
        if self.unlabeled_train_dataset is None:
            raise NameError('unlabeled_train_dataset has not been initialized!')
        logging.info(f"Getting Unlabeled Train Dataset with length {len(self.unlabeled_train_dataset)}")
        return DataLoader(
            self.unlabeled_train_dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )


    def val_dataloader(self):
        if self.val_dataset is None:
            raise NameError('val_dataset has not been initialized!')
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker
        )

 
    def test_dataloader(self):
        if self.test_dataset is None:
            raise NameError('test_dataset has not been initialized!')
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker
        )

 
    def get_unlabeled_samples(self):
        if self.unlabeled_train_dataset is None:
            raise NameError('unlabeled_train_dataset has not been initialized!')
        return self.unlabeled_train_dataset.get_samples()


    def label_sample(self, labeled_sample, unlabeled_sample_index):
        if self.unlabeled_train_dataset is None or self.labeled_train_dataset :
            raise NameError('unlabeled_train_dataset or labeled_train_dataset have not been initialized!')
        self.unlabeled_train_dataset.pop_sample(unlabeled_sample_index)
        self.labeled_train_dataset.add_sample(labeled_sample)


    def init_val_dataset(self, split_lst=None):
        pass


    def reset_val_dataset(self):
        for _ in range(
                len(self.val_dataset)
            ):
            self.labeled_train_dataset.add_sample(
                self.val_dataset.pop_sample(0)
            )


    def assign_labeled_unlabeled_split(self):
        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.initial_labeled_ratio))):
            self.unlabeled_train_dataset.add_sample(
                self.labeled_train_dataset.pop_sample(
                    random.randrange(len(self.labeled_train_dataset))
                )
            )
