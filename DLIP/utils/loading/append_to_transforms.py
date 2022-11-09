from argparse import Namespace

from DLIP.utils.data_preparation.augmentations.segmentation_augmentations import (
    ImgSegProcessingPipeline,
    SemanticSegmentationProccesor
)

def append_to_transforms(
        img_processing_args,
        train_transforms,
        val_transforms,
        test_transforms,
        val_augs=None):
    img_processing_pipeline = ImgSegProcessingPipeline(Namespace(
                                **img_processing_args))
    if val_augs is not None:
        img_processing_pipeline.make_val_aug_transform(Namespace(**val_augs))

    train_transforms.append(
        SemanticSegmentationProccesor(
            img_processing_pipeline.get_train_transform())
    )
    val_transforms.append(
        SemanticSegmentationProccesor(
            img_processing_pipeline.get_val_transform())
    )
    test_transforms.append(
        SemanticSegmentationProccesor(
            img_processing_pipeline.get_val_transform())
    )