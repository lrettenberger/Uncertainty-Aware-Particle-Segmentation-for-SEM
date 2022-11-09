import cv2

def get_border_type(keyword):
    if keyword == "constant":
        return cv2.BORDER_CONSTANT
    elif keyword == "replicate":
        return cv2.BORDER_REPLICATE
    else:
        return cv2.BORDER_REFLECT