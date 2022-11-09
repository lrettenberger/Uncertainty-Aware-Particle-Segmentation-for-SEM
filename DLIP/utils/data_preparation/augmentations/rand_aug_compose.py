import os
import cv2
import numpy as np
import albumentations as A
import random
from collections import defaultdict
from albumentations.core.composition import Transforms, Compose, BaseCompose

from albumentations.core.serialization import SERIALIZABLE_REGISTRY, instantiate_lambda

from DLIP.utils.data_preparation.augmentations.custom_augs.auto_contrast import AutoContrast
from DLIP.utils.data_preparation.augmentations.custom_augs.brightness import Brightness
from DLIP.utils.data_preparation.augmentations.custom_augs.color import Color
from DLIP.utils.data_preparation.augmentations.custom_augs.contrast import Contrast
from DLIP.utils.data_preparation.augmentations.custom_augs.cutout import Cutout
from DLIP.utils.data_preparation.augmentations.custom_augs.rotate import Rotate
from DLIP.utils.data_preparation.augmentations.custom_augs.sharpness import Sharpness
from DLIP.utils.data_preparation.augmentations.custom_augs.shear_x import ShearX
from DLIP.utils.data_preparation.augmentations.custom_augs.shear_y import ShearY
from DLIP.utils.data_preparation.augmentations.custom_augs.translate_x import TranslateX
from DLIP.utils.data_preparation.augmentations.custom_augs.translate_y import TranslateY

PARAMETER_MAX = 10 

# help functions to scale magnitude
def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

"""
    Data Augmentations
"""

def auto_contrast(**kwarg):
    return AutoContrast(always_apply=True)  

def brightness(v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return Brightness(v, always_apply=True)

def color(v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return Color(v, always_apply=True)

def contrast(v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return Contrast(v, always_apply=True)

def equalize(**kwarg):
    return A.Equalize(mode='pil', always_apply=True)

def identity(**kwarg):
    return None

def invert(**kwarg):
    return A.InvertImg(always_apply=True)

def posterize(v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return A.Posterize(v, always_apply=True)

def rotate(v, max_v, bias=0,**kwargs):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return Rotate(v, always_apply=True, **kwargs)

def sharpness(v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return Sharpness(v, always_apply=True)

def shear_x(v, max_v, bias=0,**kwargs):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return ShearX(v, always_apply=True,**kwargs)

def shear_y(v, max_v, bias=0,**kwargs):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return ShearY(v, always_apply=True,**kwargs)

def solarize(v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return A.Solarize(v, always_apply=True)

def translate_x(v, max_v, bias=0,**kwargs):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return TranslateX(v, always_apply=True,**kwargs)

def translate_y(v, max_v, bias=0,**kwargs):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return TranslateY(v, always_apply=True,**kwargs)


def fixmatch_augment_pool(
        boarder_img="replicate",
        boarder_mask="constant",
        boarder_img_value=0,
        boarder_mask_value=255
    ):
    # FixMatch paper
    boarder_dict = {
        "boarder_img": boarder_img,
        "boarder_mask": boarder_mask,
        "boarder_img_value": boarder_img_value,
        "boarder_mask_value": boarder_mask_value 
    }

    augs = [(auto_contrast, None, None, None),
            (brightness, 1.8, 0.1, None),
            (color, 1.8, 0.1, None),
            (contrast, 1.8, 0.1, None),
            (equalize, None, None, None),
            (identity, None, None, None),
            (brightness, 1.8, 0.1, None),
            (contrast, 1.8, 0.1, None),
            (equalize, None, None, None),
            (identity, None, None, None),
            (posterize, 4, 4, None),
            (rotate, 30, 0, boarder_dict),
            (sharpness, 0.9, 0.05, None),
            (shear_x, 0.3, 0, boarder_dict),
            (shear_y, 0.3, 0, boarder_dict),
            (solarize, 256, 0, None),
            (translate_x, 0.3, 0, boarder_dict),
            (translate_y, 0.3, 0, boarder_dict)]
             
    return augs

class RandAugCompose(Compose):
    """ Random Augument Compose class
    Adaption of ReplayCompose class
    https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    """
        
    def __init__(
        self,
        pre = [], 
        transforms=[], 
        n=2,
        m=10, 
        bbox_params=None, 
        keypoint_params=None, 
        additional_targets=None, 
        p=1.0, 
        save_key="replay",
        boarder_img="constant",
        boarder_mask="constant",
        boarder_img_value=0,
        boarder_mask_value=255
    ):
        if len(transforms)==0:
            self.pre    = pre
            transforms  = pre 
        
        super(RandAugCompose, self).__init__(
            transforms, 
            bbox_params, 
            keypoint_params, 
            additional_targets, 
            p
        )

        self.n = n
        self.m = m

        self.augment_pool = fixmatch_augment_pool(
            boarder_img=boarder_img,
            boarder_mask=boarder_mask,
            boarder_img_value=boarder_img_value,
            boarder_mask_value=boarder_mask_value
        )

    def __call__(self, force_apply=False, **kwargs):
        save_key = "replay"
        if not self.replay_mode:
            aug_list = []
            ops = random.choices(self.augment_pool, k=self.n)
            for op, max_v, bias, kw in ops:
                v = np.random.randint(1, self.m)
                if random.random() < 0.5:
                    if kw is not None:
                        aug_list.append(op(v=v, max_v=max_v, bias=bias,**kw))
                    else:
                        aug_list.append(op(v=v, max_v=max_v, bias=bias))

            
            aug_list.append(
                Cutout(
                    num_holes=1, 
                    max_h_size=16, 
                    max_w_size=16, 
                    fill_value=(127,127,127), 
                    always_apply=True, 
                    p=1.0
                )
            )

            transforms = self.pre + aug_list
            super(RandAugCompose, self).__init__(transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1)
            self.set_deterministic(True, save_key=save_key)
        
        self.save_key = save_key
        kwargs[self.save_key] = defaultdict(dict)
        result = super(RandAugCompose, self).__call__(force_apply=force_apply, **kwargs)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    @staticmethod
    def replay(saved_augmentations, **kwargs):
        augs = RandAugCompose._restore_for_replay(saved_augmentations)
        return augs(force_apply=True, **kwargs)

    @staticmethod
    def _restore_for_replay(transform_dict, lambda_transforms=None):
        """
        Args:
            transform (dict): A dictionary with serialized transform pipeline.
            lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
                This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
                in that dictionary should be named same as `name` arguments in respective lambda transforms from
                a serialized pipeline.
        """
        transform = transform_dict
        applied = transform["applied"]
        params = transform["params"]
        lmbd = instantiate_lambda(transform, lambda_transforms)
        if lmbd:
            transform = lmbd
        else:
            name = transform["__class_fullname__"]
            args = {k: v for k, v in transform.items() if k not in ["__class_fullname__", "applied", "params"]}
            cls = SERIALIZABLE_REGISTRY[name]
            if "transforms" in args:
                args["transforms"] = [
                    RandAugCompose._restore_for_replay(t, lambda_transforms=lambda_transforms)
                    for t in args["transforms"]
                ]
            transform = cls(**args)

        
        transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized, all_params):
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized):
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    def _to_dict(self):
        dictionary = super(RandAugCompose, self)._to_dict()
        dictionary.update({"save_key": self.save_key})
        return dictionary
