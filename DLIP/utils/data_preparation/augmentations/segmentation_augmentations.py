import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

from DLIP.utils.data_preparation.norm_std_mean import norm_std_mean

from albumentations.core.transforms_interface import ImageOnlyTransform,to_tuple

class GreyImage(ImageOnlyTransform):
    
    def __init__(self, alpha=0.5, always_apply=False, p=0.5):
            super(GreyImage, self).__init__(always_apply=always_apply,p=p)
            if isinstance(alpha, (int, float)):
                self.alpha = tuple((0.0,alpha))
            elif isinstance(alpha, (list, tuple)):
                self.alpha = alpha
            else:
                raise Exception('Error in GreyImage: alpha needs to be list,tuple, or scalar.')

    def apply(self, img, **params):
        set_alpha = np.random.uniform(self.alpha[0],self.alpha[1])
        return ((img * (1-set_alpha)) + (125 * (set_alpha))).astype(np.uint8)

    def apply_to_bbox(self, bbox, **params):
        raise Exception('not implemented')

    def apply_to_keypoint(self, keypoint, **params):
         raise Exception('not implemented')

    def get_transform_init_args_names(self):
        return ("alpha")


def list_transform(argument):
    return list(argument) if isinstance(argument,np.ndarray) else argument

class ImgSegProcessingPipeline:
    def __init__(self, params):
        self.params = params
        self.params.max_value = None
        self.trafo_dict = dict()
        self.trafo_dict["pre"] = self.make_pre_transform()
        self.trafo_dict["aug"] = self.make_aug_transform()
        self.trafo_dict["norm"] = self.make_norm_transform()
        self.trafo_dict["post"] = self.make_post_transform()
        self.trafo_dict["val_aug"] = None

    
    def make_norm_transform(self):
        norm_dict = dict()

        if (hasattr(self.params, 'norm_type')):
            norm_dict["type"] = self.params.norm_type

            if  hasattr(self.params, 'img_norm_mean') and \
                hasattr(self.params, 'img_norm_std'):
                    norm_dict["params"] = dict()
                    norm_dict["params"]["mean"] = self.params.img_norm_mean
                    norm_dict["params"]["std"] = self.params.img_norm_std
            else:
                    norm_dict["params"] = None
            return norm_dict
        return None

    def make_val_aug_transform(self, params):
        self.trafo_dict["val_aug"] = self.make_aug_transform(params=params)

    def make_pre_transform(self):
        transform = []

        if self.params.img_type == "mono_16_bit":
            self.params.max_value = 1.0
            transform.append(A.ToFloat(max_value=65535.0))
        elif self.params.img_type == "mono_8_bit":
            self.params.max_value = 1.0
            transform.append(A.ToFloat(max_value=255.0))
        elif self.params.img_type == "rgb_8_bit":
            self.params.max_value = 255.0

        if hasattr(self.params, 'img_size'):
            transform.append(
                A.Resize(height=self.params.img_size[0],
                         width=self.params.img_size[1])
            )

        return self.get_transform(transform, replay=False)

    def make_aug_transform(self,params=None):
        transform = []

        if params is None:
            params = self.params

        if hasattr(params, 'aug_flip_prob'):
            enabled=True
            if hasattr(params,'flip_enabled') and not params.flip_enabled:
                enabled=False
            if enabled:
                transform.append(A.VerticalFlip(p=params.aug_flip_prob))
                transform.append(A.HorizontalFlip(p=params.aug_flip_prob))

        if  hasattr(params, 'emboss_prob') and \
            hasattr(params, 'emboss_alpha') and \
            hasattr(params, 'emboss_strength'):
            enabled=True
            if hasattr(params,'emboss_enabled') and not params.emboss_enabled:
                enabled=False
            if enabled:
                transform.append(
                    A.IAAEmboss(
                        alpha=params.emboss_alpha,
                        p=params.emboss_prob,
                        strength=list_transform(params.emboss_strength)
                    ))
        if  hasattr(params, 'aug_gauss_noise_var_limit') and \
            hasattr(params, 'aug_gauss_noise_prob'):
                transform.append(
                    A.GaussNoise(
                        var_limit=params.aug_gauss_noise_var_limit,
                        p=params.aug_gauss_noise_prob,
                    )
                )

        if  hasattr(params, 'aug_shift_scale_rotate_shift_lim') and \
            hasattr(params, 'aug_shift_scale_rotate_scale_lim') and \
            hasattr(params, 'aug_shift_scale_rotate_rot_lim') and \
            hasattr(params, 'aug_shift_scale_rotate_prob'):
                transform.append(
                    A.ShiftScaleRotate(
                        shift_limit=params.aug_shift_scale_rotate_shift_lim,
                        scale_limit=params.aug_shift_scale_rotate_scale_lim,
                        rotate_limit=params.aug_shift_scale_rotate_rot_lim,
                        p=params.aug_shift_scale_rotate_prob,
                    )
                )

        if  hasattr(params, 'aug_rand_brightness_contrast_prob') and \
            hasattr(params, 'aug_rand_brightness_contrast_brightness_limit') and \
            hasattr(params, 'aug_rand_brightness_contrast_contrast_limit'):
                transform.append(
                    A.RandomBrightnessContrast(
                        p=params.aug_rand_brightness_contrast_prob,
                        brightness_limit=params.aug_rand_brightness_contrast_brightness_limit,
                        contrast_limit=params.aug_rand_brightness_contrast_contrast_limit,
                        )
                )

        if  hasattr(params, 'random_resized_crop_size') and \
            hasattr(params, 'random_resized_scale') and \
            hasattr(params, 'random_resized_ratio') and \
            hasattr(params, 'random_resized_propability'):
                enabled=True
                if hasattr(params,'random_resized_crop_enabled') and not params.random_resized_crop_enabled:
                    enabled=False
                if enabled:
                    transform.append(
                            A.RandomResizedCrop(
                                height=params.random_resized_crop_size[0],
                                width=params.random_resized_crop_size[1],
                                scale=params.random_resized_scale,
                                ratio=params.random_resized_ratio,
                                interpolation=cv2.INTER_CUBIC,
                                p=params.random_resized_propability
                            )
                        )

        if  hasattr(params, 'color_jitter_brightness') and \
            hasattr(params, 'color_jitter_contrast') and \
            hasattr(params, 'color_jitter_saturation') and \
            hasattr(params, 'color_jitter_hue') and \
            hasattr(params, 'color_jitter_prob'):
                enabled=True
                if hasattr(params,'color_jitter_enabled') and not params.color_jitter_enabled:
                    enabled=False
                if enabled:
                    transform.append(
                        A.ColorJitter(
                            brightness=list_transform(params.color_jitter_brightness),
                            contrast=list_transform(params.color_jitter_contrast),
                            saturation=list_transform(params.color_jitter_saturation),
                            hue=list_transform(params.color_jitter_hue),
                            p=params.color_jitter_prob
                        )
                    )

        if  hasattr(params, 'gaussian_blur_prop') and \
            hasattr(params, 'gaussian_blur_sigma_limit'):
            #hasattr(params, 'std_dev_range')
                enabled=True
                if hasattr(params,'gaussian_blur_enabled') and not params.gaussian_blur_enabled:
                    enabled=False
                if enabled:
                    transform.append(
                        A.GaussianBlur(
                            p=params.gaussian_blur_prop,
                            blur_limit = 0,
                            sigma_limit=params.gaussian_blur_sigma_limit
                        )
                    )  
                    
        if  hasattr(params, 'gray_image_alpha') and \
            hasattr(params, 'gray_image_prop'):
                transform.append(
                    GreyImage(
                        p=params.gray_image_prop,
                        alpha=params.gray_image_alpha,
                    )
                )  

        if  hasattr(params, 'to_gray_prob'):
                enabled=True
                if hasattr(params,'to_grayscale_enabled') and not params.to_grayscale_enabled:
                    enabled=False
                if enabled:
                    transform.append(
                        A.ToGray(
                            p= params.to_gray_prob
                        )
                    ) 

        if  hasattr(params, 'solarization_prop'):
                enabled=True
                if hasattr(params,'solarization_enabled') and not params.solarization_enabled:
                    enabled=False
                if enabled:
                    threshold = 128
                    if hasattr(params, 'solarization_threshold'):
                        threshold = list_transform(params.solarization_threshold)
                    transform.append(
                        A.Solarize(
                            p=params.solarization_prop,
                            threshold = threshold
                        )
                    )
    

        if  hasattr(params, 'val_augs'):
            if params.val_augs == True:
                    transform.append(
                        A.RandomResizedCrop(
                            height=params.val_augs_size[0],
                            width=params.val_augs_size[1],
                            scale=[0.7, 1.0],
                            ratio=[1, 1],
                            interpolation=cv2.INTER_CUBIC,
                            p=0.5
                        )
                    )
                    transform.append(A.VerticalFlip(p=0.5))
                    transform.append(A.HorizontalFlip(p=0.5))
            else:
                    transform.append(
                        A.Resize(
                            height=params.post_augmentation_resize_size[0],
                            width=params.post_augmentation_resize_size[1]
                        )
                    ) 
                    transform.append(
                        A.CenterCrop(
                            height=params.center_crop_size[0],
                            width=params.center_crop_size[1]
                        )
                    )
                

        return self.get_transform(
                transform, replay=params.replay_processing_pipeline if hasattr(params, 'replay_processing_pipeline') else False
            )

    def make_post_transform(self):
        transform = []

        transform.append(A.ToFloat())
        transform.append(ToTensorV2())

        return self.get_transform(transform, replay=False)

    def get_transform(self, trafo_lst, replay=False):
        if replay:
            return A.ReplayCompose(trafo_lst)
        else:
            return A.Compose(trafo_lst)

    def get_val_transform(self):
        trafo_dict = self.trafo_dict.copy()
        trafo_dict["aug"] = None
        if self.trafo_dict["val_aug"] is not None:
             trafo_dict["aug"] = self.trafo_dict["val_aug"]
        return trafo_dict

    def get_train_transform(self):
        trafo_dict = self.trafo_dict.copy()
        return trafo_dict

    def get_pre_transform(self):
        trafo_dict = self.trafo_dict.copy()
        trafo_dict["aug"] = None
        trafo_dict["post"] = None
        return trafo_dict


class SemanticSegmentationProccesor:
    def __init__(self, transform: dict()):
        if ("pre" not in transform or
                "aug" not in transform or
                "post" not in transform):
            raise Exception("Wrong dict structure")
        self.transform = transform

    def __call__(self, image, mask=None):
        if mask is None:
            mask = np.zeros((image.shape))
        transformed_data = self.transform["pre"](image=image, mask=mask)
        sample_img, label_one_hot = (
            transformed_data["image"],
            transformed_data["mask"],
        )

        mean, std = np.mean(sample_img, axis=(0,1)), np.std(sample_img, axis=(0,1))

        if self.transform["aug"] is not None:
            transformed_data = self.transform["aug"](
                image=sample_img, mask=label_one_hot
            )
            sample_img, label_one_hot, transformations = (
                transformed_data["image"],
                transformed_data["mask"],
                transformed_data["replay"] if "replay" in transformed_data else []
            )
        else:
            transformations = []

        if self.transform["norm"] is not None:
            if self.transform["norm"]["type"]=="per_image":
                sample_img = norm_std_mean(sample_img)
            elif self.transform["norm"]["type"]=="per_dataset":
                sample_img = norm_std_mean(
                    sample_img, 
                    mean=self.transform["norm"]["params"]["mean"],
                    std=self.transform["norm"]["params"]["std"])
        
        if self.transform["post"] is not None:
            transformed_data = self.transform["post"](
                image=sample_img, mask=label_one_hot
            )
            sample_img, label_one_hot = (
                transformed_data["image"],
                transformed_data["mask"],
            )
        return sample_img, label_one_hot, transformations
