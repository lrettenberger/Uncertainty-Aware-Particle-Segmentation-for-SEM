from DLIP.data.base_classes.segmentation.base_seg_dataset import BaseSegmentationDataset

class SoftLabelDataset(BaseSegmentationDataset):
    
    def __init__(
        self,
        root_dir: str,
        samples_dir: str = "samples",
        class_dir: str = "labels",
        samples_data_format="tif",
        labels_data_format="tif",
        transforms=None, 
        empty_dataset=False,
        labels_available=True,
        return_trafos=False,
        label_transform=None
    ):
        super().__init__(
            root_dir,
            None,
            samples_dir=samples_dir,
            class_dir=class_dir,
            samples_data_format=samples_data_format,
            labels_data_format=labels_data_format,
            transforms=transforms,
            empty_dataset=empty_dataset,
            labels_available=labels_available,
            return_trafos=return_trafos,
            label_transform=label_transform,
        )
        
