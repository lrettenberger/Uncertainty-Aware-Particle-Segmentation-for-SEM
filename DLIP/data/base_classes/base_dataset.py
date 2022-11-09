"""
    The base dataset. A new dataset should extend this base class, to ensure full flexibility
    and usability. Since the datasets can potentially only be partially labeled, the implementation
    needs to provide functions to add, pop (remove) and get all samples of the dataset.
    A raw mode should also be suported, in which the raw samples of the dataset should be returned.
    An example implementation can be found at: DLIP.data.example_dataset.example_dataset.py     

"""
from abc import abstractmethod
from typing import List
from torch.utils.data import Dataset
import logging


class BaseDataset(Dataset):
    
    def __init__(self):
        super().__init__()
        # The list of indices for this dataset. 
        # Should be set by the implementing class.
        self.indices: List = None
    
    @abstractmethod
    def get_samples(self):
        raise NotImplementedError("This method needs to be implemented.")

    @abstractmethod
    def pop_sample(self, index):
        raise NotImplementedError("This method needs to be implemented.")

    @abstractmethod
    def add_sample(self, new_sample):
        raise NotImplementedError("This method needs to be implemented.")

    def set_raw_mode(self,raw_mode):
        self.raw_mode = raw_mode

    def resort_samples(self):
        try:
            self.indices = sorted(self.indices,key=lambda x: int(x))
        except:
            logging.warn("Sorting not possible due to no integer names")
