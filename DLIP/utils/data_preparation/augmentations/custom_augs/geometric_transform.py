import albumentations as A


class GeometricTransform(A.DualTransform):
    """Base Geometric Transform"""
    def __init__(
        self, 
        value, 
        boarder_img="reflect", 
        boarder_mask="constant", 
        boarder_img_value=0,
        boarder_mask_value=255,
        always_apply=False, 
        p=0.5):
        super(GeometricTransform, self).__init__(always_apply, p)
        self.value = value
        self.boarder_img = boarder_img
        self.boarder_mask = boarder_mask
        self.boarder_img_value = boarder_img_value
        self.boarder_mask_value = boarder_mask_value

    def get_params(self):
        return {
            "value": self.value,
            "boarder_img": self.boarder_img,
            "boarder_mask": self.boarder_mask,
            "boarder_img_value": self.boarder_img_value,
            "boarder_mask_value": self.boarder_mask_value
        }

    def get_transform_init_args_names(self):
        return ("value","boarder_img","boarder_mask", "boarder_img_value", "boarder_mask_value",)    
