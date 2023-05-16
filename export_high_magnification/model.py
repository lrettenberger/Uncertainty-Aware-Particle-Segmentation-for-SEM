import numpy as np
import cv2

OVERLAPPING_THRESHOLD = 0.3
MIN_ACTIVATION = 0.40
THRESHOLD_CONF = 0.95
THRESHOLD_NOT_CONF = 0.6
CONF_NOT_CONF_OVERLAP = 0.1


def conf_not_conf_exclusive(masks_conf,masks_not_conf):
    for i in np.unique(masks_conf):
        for j in np.unique(masks_not_conf):
            if i==0 or j==0:
                continue
            intersection = np.logical_and((masks_conf==i)*1, (masks_not_conf==j)*1)
            union = np.logical_or((masks_conf==i)*1, (masks_not_conf==j)*1)
            IOU_SCORE = np.sum(intersection) / np.sum(union)
            if IOU_SCORE >= CONF_NOT_CONF_OVERLAP:
                masks_not_conf[masks_not_conf==j] = 0
    return masks_conf,masks_not_conf

def post_process_preds(ort_outs,lower_threshold,high_threshold,overlapping_threshold,min_activation,H,W):
    scores = ort_outs[2]
    masks = ort_outs[3]
    masks = masks[(scores > lower_threshold) & (scores <= high_threshold)]
    scores = scores[(scores > lower_threshold) & (scores <= high_threshold)]
    final_mask = np.zeros((H, W), dtype=np.float32)
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
            mask[mask > min_activation] = local_instance_number
            mask[mask < min_activation] = 0
            if np.sum(mask) > 0:
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
    B,C,H,W = x.shape
    preds_confident = post_process_preds(ort_outs,THRESHOLD_CONF,1.0,OVERLAPPING_THRESHOLD,MIN_ACTIVATION,H,W)
    preds_not_confident = post_process_preds(ort_outs,THRESHOLD_NOT_CONF,THRESHOLD_CONF,OVERLAPPING_THRESHOLD,MIN_ACTIVATION,H,W)
    preds_combined = post_process_preds(ort_outs,THRESHOLD_NOT_CONF,1.0,OVERLAPPING_THRESHOLD,MIN_ACTIVATION,H,W)
    
    for i in range(len(preds_confident)):
        mask_conf_i, mask_not_conf_i = conf_not_conf_exclusive(preds_confident[i],preds_not_confident[i])
        preds_confident[i] = mask_conf_i
        preds_not_confident[i] = mask_not_conf_i
    
    return preds_not_confident, preds_confident, preds_combined

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
    if image.shape[0] < patch_size or image.shape[1] < patch_size:
        # pad image with zeros if its smaller than patch_size to be able to process it
        padded_image = np.zeros((patch_size,patch_size))
        padded_image[:image.shape[0],:image.shape[1]] = image
        image = padded_image
    mask_not_confident, mask_confident, mask_combined = get_mask(image,ort_session)
    return mask_not_confident, mask_confident, mask_combined