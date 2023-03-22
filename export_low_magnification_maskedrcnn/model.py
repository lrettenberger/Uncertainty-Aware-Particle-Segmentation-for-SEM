import numpy as np
import cv2

OVERLAPPING_THRESHOLD = 0.41
MIN_ACTIVATION = 0.53
THRESHOLD_CONF = 0.8
MASK_FILTER_THRESHOLD = 0.005

def post_process_preds(ort_outs,overlapping_threshold,min_activation,lower_threshold,high_threshold):
    scores = ort_outs[2]
    masks = ort_outs[3]
    C,H,W = masks[0].shape
    masks = masks[(scores > lower_threshold) & (scores <= high_threshold)]
    scores = scores[(scores > lower_threshold) & (scores <= high_threshold)]
    final_mask = np.zeros((H,W), dtype=np.float32)
    local_instance_number = 1    
    for i in range(len(masks)):
        if scores[i] == 0:
            continue
        for j in range(i+1, len(masks)):
            if scores[j] == 0:
                continue
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
            if np.sum(mask)/np.size(mask) > MASK_FILTER_THRESHOLD:
                # only if the mask is big enough
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
    preds = post_process_preds(ort_outs,OVERLAPPING_THRESHOLD,MIN_ACTIVATION,THRESHOLD_CONF,1.0)
    return preds

def get_segmentation_mask(image,ort_session):
    mask = get_mask(image,ort_session)
    return mask