import cv2
import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import torchvision.transforms.functional as torchvision

from DLIP.utils.data_preparation.augmentations.custom_augs.geometric_transform import GeometricTransform
from DLIP.utils.data_preparation.augmentations.custom_augs.get_boarder_type import get_border_type


class Rotate(GeometricTransform):
    """Rotate Transform"""
    def __init__(
        self, 
        value, 
        boarder_img="reflect", 
        boarder_mask="constant", 
        boarder_img_value=0,
        boarder_mask_value=255,
        always_apply=False, 
        p=0.5):
        super(Rotate, self).__init__(
            value, 
            boarder_img, 
            boarder_mask, 
            boarder_img_value, 
            boarder_mask_value,
            always_apply,
            p)

    def apply(        
        self, 
        image, 
        value=0, 
        boarder_img="reflect", 
        boarder_img_value=0,
        **params
        ):
        boarder_type = get_border_type(boarder_img)
        boarder_value = boarder_img_value*np.ones(image.shape[2])

        img_shape = image.shape
        padding = max(img_shape[0], img_shape[1])
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, boarder_type,value=boarder_value)
        image = PIL.Image.fromarray(image)
        image = image.rotate(angle=value, resample=PIL.Image.BILINEAR) 
        image = torchvision.resized_crop(img=image, top=padding, left=padding, height=img_shape[0],
                                                        width=img_shape[1], size=img_shape[0:2], interpolation=2)

        return np.asarray(image, dtype=np.uint8)

    def apply_to_mask(
        self, 
        image, 
        value=0, 
        boarder_mask="reflect", 
        boarder_mask_value=255,
        **params
        ):
        input_type = image.dtype
        
        boarder_type = get_border_type(boarder_mask)
        boarder_value = boarder_mask_value*np.ones(image.shape[2])

        result = np.zeros_like(image)
        img_shape = image.shape
        padding = max(img_shape[0], img_shape[1])
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, boarder_type, value=boarder_value)

        for i in range(image.shape[2]):
            mask_i = PIL.Image.fromarray(image[:,:,i])
            mask_i = mask_i.rotate(angle=value, resample=PIL.Image.NEAREST) 
            mask_i = torchvision.resized_crop(img=mask_i, top=padding, left=padding, height=img_shape[0],
                                             width=img_shape[1], size=img_shape[0:2], interpolation=0)
            result[:,:,i] = np.asarray(mask_i, dtype=input_type)

        return result
