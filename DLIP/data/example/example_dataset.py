"""
    Example implementation of a dataset. Does not furfill any purpose apart from providing guidance.
"""
import numpy as np
from DLIP.data.base_classes.base_dataset import BaseDataset


class ExampleDataset(BaseDataset):
    
    def __init__(self, num_of_images: int, labels_available: bool):
        super().__init__()
        self.labels_available = labels_available
        
        # If the number of images is 0 (or smaller) the dataset is empty
        if num_of_images <= 0:
            # An empty dataset does not contain anything
            self.indices = None
            self.images = None
            if self.labels_available:
                self.labels = np.zeros((2,0))
        else:
            # create an variable self.images which contains ''images'' of size (3,256,256)
            # which either contain only ones or zeros
            ones = np.float32(np.ones((num_of_images//2,3, 256, 256)))
            zeros = np.float32(np.zeros((num_of_images//2,3, 256, 256)))
            self.images = np.float32(np.concatenate((ones,zeros),axis=0))
            np.random.shuffle(self.images)
            # In this simple case the indices are just the indices of the numpy array of the images
            # for more complex cases (e.g. if the images are directory entries) the indices should provide
            # a way to handle the samples of the dataset with integer indices.
            self.indices = np.linspace(0,len(self.images)-1,num=len(self.images)).astype(np.int)
            # The indices are shuffled, to generate at least some sense for the indices list.
            np.random.shuffle(self.indices)
            # If labels should be available  they are initaialized
            if self.labels_available:
                # In this case the labels should describe whether an inidividual image contains zeros or ones
                # The label should be 1 if it conatins ones and 0 if it contains zeros
                self.labels = np.float32(np.expand_dims(np.array([int(np.max(x)) for x in self.images]),1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # return the correct image at the index. Note that the images variable is not accesed directly, but over the
        # indices array, since it holds the correct (shuffeled) indices.
        index = self.indices[idx]
        if self.labels_available:
            return (self.images[index],self.labels[index])
        return np.astype(self.images[index],np.float32)

    def get_samples(self):
        return self.images 


    def pop_sample(self, idx):
        popped = self[idx]
        self.indices = np.delete(self.indices, idx, axis=0)
        return popped


    def add_sample(self, new_sample):
        # Here we need to make some distinctions between labels_available = True and
        # labels_available = False and if the variables have already been initialized.
        # Not very important for understanding the concept.
        sample = new_sample
        if self.labels_available:
            sample, label = new_sample
            self.labels = np.float32(np.expand_dims(np.append(self.labels, label),1))
        if self.images is None:
            self.images = np.expand_dims(sample, 0)
        else:
            self.images = np.concatenate((self.images,np.expand_dims(sample,0)),axis=0)
        if self.indices is None:
            self.indices = [0]
        else:
            self.indices.append(max(self.indices)+1) 
