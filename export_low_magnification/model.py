import numpy as np
import cv2

def get_mask(x,ort_session):
    x = np.expand_dims(np.expand_dims(x.astype(np.float32)/255.,0),0)
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0][0,0]

# adapted from https://github.com/obss/sahi/blob/e798c80d6e09079ae07a672c89732dd602fe9001/sahi/slicing.py#L30, MIT License
def calculate_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.5,
    overlap_width_ratio: float = 0.5,
):
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.
    :param image_height: Height of the original image.
    :param image_width: Width of the original image.
    :param slice_height: Height of each slice
    :param slice_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of bounding boxes in xyxy format
    """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def get_segmentation_mask(image,ort_session,blur_kernel_size,patch_size=1024):
    original_size = image.shape
    image = cv2.resize(image,(round(image.shape[1]*(1024/1200)),round(image.shape[0]*(1024/1200))),cv2.INTER_NEAREST)
    rezized_size = image.shape
    if image.shape[0] < patch_size or image.shape[1] < patch_size:
        # pad image with zeros if its smaller than patch_size to be able to process it
        padded_image = np.zeros((patch_size,patch_size))
        padded_image[:image.shape[0],:image.shape[1]] = image
        image = padded_image
    boxes = calculate_slice_bboxes(image.shape[0],image.shape[1],patch_size,patch_size)
    image_occurences_map = np.zeros_like(image)
    combined_mask = np.zeros_like(image).astype(np.float32)
    for box in boxes:
        x1,y1,x2,y2 = box
        image_occurences_map[y1:y2,x1:x2] = image_occurences_map[y1:y2,x1:x2]+1
        mask = get_mask(image[y1:y2,x1:x2],ort_session)
        combined_mask[y1:y2,x1:x2] = combined_mask[y1:y2,x1:x2]+mask
    mask = combined_mask / image_occurences_map
    mask = cv2.GaussianBlur(mask,(blur_kernel_size,blur_kernel_size),0)
    mask = mask[:rezized_size[0],:rezized_size[1]]
    return cv2.resize(mask,original_size[::-1],cv2.INTER_NEAREST)