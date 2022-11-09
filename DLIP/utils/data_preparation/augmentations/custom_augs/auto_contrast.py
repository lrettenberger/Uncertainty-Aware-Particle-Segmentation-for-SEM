import numpy as np
import albumentations as A
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps


class AutoContrast(A.ImageOnlyTransform):
    """PIL.ImageOps.autocontrast Transform"""

    def apply(self, img, **params):
        img = PIL.Image.fromarray(img, mode='RGB')
        return np.asarray(PIL.ImageOps.autocontrast(img), dtype=np.uint8)
    
    def get_transform_init_args_names(self):
        return ()
