import numpy as np
import cv2

OVERLAPPING_THRESHOLD = 0.3
MIN_ACTIVATION = 0.45

def post_process_preds(ort_outs,overlapping_threshold,min_activation):
    scores = ort_outs[2]
    masks = ort_outs[3]
    score_threshold = 0.5
    scores = scores[ort_outs[2] > score_threshold]
    masks = masks[ort_outs[2] > score_threshold]
    final_mask = np.zeros((1024, 1638), dtype=np.float32)
    local_instance_number = 1    
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            intersection = np.logical_and(masks[i], masks[j])
            union = np.logical_or(masks[i], masks[j])
            IOU_SCORE = np.sum(intersection) / np.sum(union)
            if IOU_SCORE > overlapping_threshold:
                scores[j] = 0
    for mask, score in zip(masks, scores):
            # as scores is already sorted        
            if score == 0:
                continue        
            mask = mask.squeeze()
            mask[mask > min_activation] = local_instance_number
            mask[mask < min_activation] = 0
            local_instance_number += 1
            temp_filter_mask = np.where(final_mask > 1, 0., 1.)
            temp_filter_mask = (final_mask < 1)*1.
            mask = mask * temp_filter_mask        
            final_mask += mask    
    return final_mask

def get_mask(x,ort_session):
    x = np.expand_dims(np.expand_dims(x.astype(np.float32)/255.,0),0)
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    preds = post_process_preds(ort_outs,OVERLAPPING_THRESHOLD,MIN_ACTIVATION)
    return preds

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


def get_segmentation_mask(image,ort_session,patch_size=1024):
    original_size = image.shape
    # scale down
    image = cv2.resize(image,(round(image.shape[1]*(1024/1200)),round(image.shape[0]*(1024/1200))), interpolation = cv2.INTER_NEAREST)
    if image.shape[0] < patch_size or image.shape[1] < patch_size:
        # pad image with zeros if its smaller than patch_size to be able to process it
        padded_image = np.zeros((patch_size,patch_size))
        padded_image[:image.shape[0],:image.shape[1]] = image
        image = padded_image
    mask = get_mask(image,ort_session).astype(np.int16)
    return cv2.resize(mask,original_size[::-1], interpolation = cv2.INTER_NEAREST)