import numpy as np
from DLIP.utils.metrics.inst_seg_metrics import remap_label

def get_generic_inv_agreement(mcd_results, get_metric):
    T = mcd_results.shape[0]
    M = 0
    for i in range(T):
        for j in range(T):
            if j > i:
                metric = get_metric(remap_label(mcd_results[i,:]),remap_label(mcd_results[j,:]))
                M = M + metric*(i!=j) 
    score = 1 - 2/(T*(T-1)) * M 
    return score