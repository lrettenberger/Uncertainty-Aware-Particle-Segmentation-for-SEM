"""
    Datasets to be used must be specified here to be loadable.
"""
from .example import ExamplePLDatamodule
from .example import ExampleDataset
from .soft_label_synth import SoftLabelDataset, SoftLabelDataModule
from .base_classes.segmentation.base_seg_data_module import GenericSegmentationDataModule