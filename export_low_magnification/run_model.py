import onnxruntime
import cv2
import argparse
import urllib.request
from tqdm import tqdm
import os
from pathlib import Path
from glob import glob
import numpy as np
import tifffile

from dist_map_post_processor import DistMapPostProcessor
from utils import load_image_from_path, write_parameters_to_file
from inst_seg_contour import visualize_instances_map
from model import get_segmentation_mask

# download model if its not present
if not os.path.isfile("model_low_mag_maskedrcnn.onnx"):
    print('Downloading Model...')
    urllib.request.urlretrieve("https://bwsyncandshare.kit.edu/s/o929J84orZr4N6t/download/model_low_mag_maskedrcnn.onnx", "model_low_mag_maskedrcnn.onnx")

VALID_IMAGE_FORMATS = ['.png','.tiff','.tif']

# The neural network
ort_session = onnxruntime.InferenceSession("model_low_mag_maskedrcnn.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

parser = argparse.ArgumentParser(description='Calculate Instance Segmentations to SEM Images.')
parser.add_argument(
    '--input',
    dest='input',
    type=str,
    default='./images',
    help='If the path is an image, the image will be processed. If its a path, the whole directory will be processed.'
)
args = parser.parse_args()
input = args.input

input_images = []
if os.path.isdir(input):
    for image_format in VALID_IMAGE_FORMATS:
        input_images = input_images + glob(f'{input}/*{image_format}')
else:
    input_images = [input]
   
pbar = tqdm(input_images) 
for input_image_path in pbar:
    image_name = ''.join(input_image_path.split('/')[-1]).split('.')[0]
    output_path = '/'.join(input_image_path.split('/')[:-1] + ['out'])
    Path(output_path).mkdir(parents=True, exist_ok=True)
    pbar.set_description(f"Processing {image_name}")

    # Load image
    image = load_image_from_path(input_image_path)

    # Get segmentation mask
    mask_not_confident, mask_confident, mask_combined = get_segmentation_mask(image,ort_session)
    tifffile.imwrite(f'{output_path}/{image_name.replace(".png","")}_uint16_instance_mask_not_confident.tiff',mask_not_confident.astype(np.int16))
    tifffile.imwrite(f'{output_path}/{image_name.replace(".png","")}_uint16_instance_mask_confident.tiff',mask_confident.astype(np.int16))
    tifffile.imwrite(f'{output_path}/{image_name.replace(".png","")}_uint16_instance_mask_combined.tiff',mask_combined.astype(np.int16))