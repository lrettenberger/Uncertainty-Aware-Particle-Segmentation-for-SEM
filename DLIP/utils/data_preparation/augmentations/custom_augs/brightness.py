import numpy as np
import albumentations as A
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps


class Brightness(A.ImageOnlyTransform):
    """PIL.ImageEnhance.Brightness Transform"""
    def __init__(self, value, always_apply=False, p=0.5):
        super(Brightness, self).__init__(always_apply, p)
        self.value = value

    def apply(self, image, value=0, **params):
        image = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageEnhance.Brightness(image).enhance(value), dtype=np.uint8)

    def get_params(self):
        return {
            "value": self.value
        }

    def get_transform_init_args_names(self):
        return ("value",)
