import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything
from tqdm import tqdm
from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label
import cv2
import numpy as np
import onnxruntime
from glob import glob
import tifffile

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


def get_segmentation_mask(image,ort_session,patch_size=1024):
    boxes = calculate_slice_bboxes(image.shape[0],image.shape[1],patch_size,patch_size)
    image_occurences_map = np.zeros_like(image)
    combined_mask = np.zeros_like(image).astype(np.float32)
    for box in boxes:
        x1,y1,x2,y2 = box
        image_occurences_map[y1:y2,x1:x2] = image_occurences_map[y1:y2,x1:x2]+1
        mask = get_mask(image[y1:y2,x1:x2],ort_session)
        combined_mask[y1:y2,x1:x2] = combined_mask[y1:y2,x1:x2]+mask
    return combined_mask / image_occurences_map


ort_session = onnxruntime.InferenceSession("model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

i=0
th_seed_start = 0.1
th_seed_end = 0.99
th_seed_step_size = 0.05 # 0.05
th_cell_start = 0.01
th_cell_step_size = 0.002 # 0.002
total_num = sum([len(np.arange(th_cell_start,x,th_cell_step_size)) for x in np.arange(th_seed_start,th_seed_end,th_seed_step_size)])*6
f = open("hyperparam_results_whole_images.txt", "a")
for blur in tqdm([0,3,9,11,15,19],leave=False):
    masks = []
    labels = []
    for image_path in (glob('/home/ws/kg2371/datasets/sem_segmentation_cleaned_round_3/test/full_size_samples/*')):
        image = tifffile.imread(image_path)
        image = cv2.resize(image,(round(image.shape[1]*(1024/1200)),round(image.shape[0]*(1024/1200))))
        label = tifffile.imread(image_path.replace('samples','labels'))
        label = cv2.resize(label,(round(label.shape[1]*(1024/1200)),round(label.shape[0]*(1024/1200))))
        mask = get_segmentation_mask(image,ort_session)
        if blur > 0:
            mask = cv2.GaussianBlur(mask,(blur,blur),0)
        masks.append(mask)
        labels.append(label)
    for do_splitting in [False]:
        for do_fill_holes in [False]:
            for th_seed in tqdm(np.arange(th_seed_start,th_seed_end,th_seed_step_size),leave=False):
                for th_cell in tqdm(np.arange(th_cell_start,th_seed,th_cell_step_size),leave=False):
                    post_pro = DistMapPostProcessor(
                        sigma_cell=1.0,
                        th_cell=th_cell,
                        th_seed=th_seed,
                        do_splitting=do_splitting,
                        do_area_based_filtering=False,
                        do_fill_holes=do_fill_holes,
                        valid_area_median_factors=[0.25,3]
                    )
                    ajis=[]
                    for j in range(len(masks)):
                        ajis.append(get_fast_aji_plus(remap_label(labels[j]),remap_label(post_pro.process(masks[j],None))))
                    ajis = np.array(ajis)
                    i+=1
                    print(f'Progress: {(i/total_num)*100:.2f}%')
                    f.write(f'{blur},{do_splitting},{do_fill_holes},{th_seed},{th_cell},{np.mean(ajis)}\n')
                    f.flush()
f.close()