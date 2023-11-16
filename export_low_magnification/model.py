import numpy as np
import cv2

IMG_SIZE = (1728,1080)

MIN_ACTIVATION = 0.4148937301911166
THRESHOLD_CONF = 0.7450274476935512
THRESHOLD_NOT_CONF = 0.6997302456756461
OVERLAPPING_THRESHOLD = 0.7041961899541517
CONF_NOT_CONF_OVERLAP = 0.4935919386586958

MIN_PARTICLE_SIZE_PIXELS = 1000
NUM_BINS = 6

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


# 0 boxes
# 1 labels
# 2 scores
# 3 sem scores
# 4 masks

def post_process_preds(ort_outs,mode='both',H=0,W=0):
    scores = ort_outs[2]
    sem_scores = ort_outs[3]
    sem_scores = np.argmax(sem_scores,axis=1)
    masks = ort_outs[4]
    if mode=='conf':
        masks = masks[(sem_scores  >= (NUM_BINS/2))]
        scores = scores[(sem_scores >= (NUM_BINS/2))]
        masks = masks[(scores > THRESHOLD_CONF)]
        scores = scores[(scores > THRESHOLD_CONF)]
    elif mode=='not conf':
        masks = masks[(sem_scores < (NUM_BINS/2))]
        scores = scores[(sem_scores  < (NUM_BINS/2))]
        masks = masks[(scores > THRESHOLD_NOT_CONF)]
        scores = scores[(scores > THRESHOLD_NOT_CONF)]
    else:
        masks = masks[(scores > THRESHOLD_NOT_CONF)]
        scores = scores[(scores > THRESHOLD_NOT_CONF)]
    final_mask = np.zeros((H,W))*1.
    local_instance_number = 1
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            intersection = np.logical_and(masks[i], masks[j])
            union = np.logical_or(masks[i], masks[j])
            IOU_SCORE = np.sum(intersection) / np.sum(union)
            if IOU_SCORE > OVERLAPPING_THRESHOLD:
                scores[j] = 0
    for mask, score in zip(masks, scores):
        # as scores is already sorted        
        if score == 0:
            continue
        mask = mask.squeeze().copy()
        mask[mask > MIN_ACTIVATION] = local_instance_number        
        mask[mask < MIN_ACTIVATION] = 0
        if np.sum(mask) > 0:
            local_instance_number += 1       
            temp_filter_mask = np.where(final_mask > 1, 0., 1.)
            mask = mask * temp_filter_mask  
            final_mask += mask
    return final_mask

def get_mask(x,ort_session):
    x = np.expand_dims(np.expand_dims(x.astype(np.float32)/255.,0),0)
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    B,C,H,W = x.shape
    preds_confident = post_process_preds(ort_outs,'conf',H,W)
    preds_not_confident = post_process_preds(ort_outs,'not conf',H,W)
    preds_combined = post_process_preds(ort_outs,'both',H,W)
    
    for i in range(len(preds_confident)):
        mask_conf_i, mask_not_conf_i = conf_not_conf_exclusive(preds_confident[i],preds_not_confident[i])
        preds_confident[i] = mask_conf_i
        preds_not_confident[i] = mask_not_conf_i    
    return preds_not_confident, preds_confident, preds_combined

def area_based_filtering(masks):
    for i in np.unique(masks):
        if i==0:
            # background
            continue
        if np.sum(masks==i) <= MIN_PARTICLE_SIZE_PIXELS:
            masks[masks==i] = 0
            masks[masks>=i]-=1
    return masks
            


def get_segmentation_mask(image,ort_session,patch_size=1024):
    if image.shape[0] < patch_size or image.shape[1] < patch_size:
        # pad image with zeros if its smaller than patch_size to be able to process it
        padded_image = np.zeros((patch_size,patch_size))
        padded_image[:image.shape[0],:image.shape[1]] = image
        image = padded_image
    mask_not_confident, mask_confident, mask_combined = get_mask(cv2.resize(image,IMG_SIZE,interpolation=cv2.INTER_LINEAR),ort_session)
    mask_not_confident = area_based_filtering(cv2.resize(mask_not_confident,image.shape[::-1],interpolation=cv2.INTER_NEAREST))
    mask_confident = area_based_filtering(cv2.resize(mask_confident,image.shape[::-1],interpolation=cv2.INTER_NEAREST))
    mask_combined = area_based_filtering(cv2.resize(mask_combined,image.shape[::-1],interpolation=cv2.INTER_NEAREST))
    return mask_not_confident, mask_confident, mask_combined