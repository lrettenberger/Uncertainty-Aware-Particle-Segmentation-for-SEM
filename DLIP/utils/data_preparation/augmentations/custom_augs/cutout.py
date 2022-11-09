import albumentations as A


class Cutout(A.Cutout):
    """ adapted albumentations Cutout Transform
    bug in implementation, doesn't save color values"""
    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size", "fill_value")
