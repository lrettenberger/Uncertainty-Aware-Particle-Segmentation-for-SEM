import cv2
import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import torchvision.transforms.functional as torchvision
import random

from DLIP.utils.data_preparation.augmentations.custom_augs.geometric_transform import GeometricTransform
from DLIP.utils.data_preparation.augmentations.custom_augs.get_boarder_type import get_border_type


class TranslateX(GeometricTransform):
    """TranslateX Transform"""
    def __init__(
        self, 
        value, 
        boarder_img="reflect", 
        boarder_mask="constant", 
        boarder_img_value=0,
        boarder_mask_value=255,
        random_sign=False,
        always_apply=False, 
        p=0.5):
        super(TranslateX, self).__init__(
            value, 
            boarder_img, 
            boarder_mask, 
            boarder_img_value, 
            boarder_mask_value,
            always_apply,
            p)
        self.random_sign = random_sign

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
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, boarder_type, value=boarder_value)
        image = PIL.Image.fromarray(image, mode='RGB')
        value = int(value * img_shape[0])
        image = image.transform(image.size, PIL.Image.AFFINE, (1, 0, value, 0, 1, 0), resample=PIL.Image.BILINEAR) 
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

        value = int(value * img_shape[0])
        image = cv2.copyMakeBorder(image, padding, padding, padding, padding, boarder_type, value=boarder_value)

        for i in range(image.shape[2]):
            mask_i = PIL.Image.fromarray(image[:,:,i])
            
            mask_i = mask_i.transform(mask_i.size, PIL.Image.AFFINE, (1, 0, value, 0, 1, 0), resample=PIL.Image.NEAREST) 
            mask_i = torchvision.resized_crop(img=mask_i, top=padding, left=padding, height=img_shape[0],
                                             width=img_shape[1], size=img_shape[0:2], interpolation=0)
            result[:,:,i] = np.asarray(mask_i, dtype=input_type)

        return result

    def get_params(self):
        sign = 1
        if self.random_sign:
            if random.random() < 0.5:
                sign = -1

        return {
            "value": self.value*sign,
            "boarder_img": self.boarder_img,
            "boarder_mask": self.boarder_mask,
            "boarder_img_value": self.boarder_img_value,
            "boarder_mask_value": self.boarder_mask_value
        }
