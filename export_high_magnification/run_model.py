import onnxruntime
import cv2
import tifffile
import argparse
from tqdm import tqdm
import urllib.request 
import os
from pathlib import Path
from glob import glob
import numpy as np

from dist_map_post_processor import DistMapPostProcessor
from utils import load_image_from_path, write_parameters_to_file
from inst_seg_contour import visualize_instances_map
from model import get_segmentation_mask


# download model if its not present
print('Downloading Model...')
urllib.request.urlretrieve("https://bwsyncandshare.kit.edu/s/NH2qNF5pTDyGReR/download/model_high_mag_maskedrcnn.onnx", "model_high_mag_maskedrcnn.onnx")



VALID_IMAGE_FORMATS = ['.png','.tiff','.tif']

# Instance segmentation post processing
BLUR_KERNEL_SIZE = 9
TH_CELL=0.022,
TH_SEED=0.25,
post_pro = DistMapPostProcessor(
    sigma_cell=1.0,
    th_cell=TH_CELL,
    th_seed=TH_SEED,
    do_splitting=False,
    do_area_based_filtering=False,
    do_fill_holes=False,
    valid_area_median_factors=[0.25,3]
)
# The neural network
ort_session = onnxruntime.InferenceSession("./model_high_mag_maskedrcnn.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

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
    # save orig size for output
    original_size = image.shape
    # low pass filter to get high frequencies smoothed out
    kernel = np.ones((3,3),np.float32)/9
    image = cv2.filter2D(image,-1,kernel)
    # get to right dims
    image = cv2.resize(image,(1920,1200),interpolation=cv2.INTER_NEAREST)
    # Get resize factor, assuming that the image had a size of (resized_factor*1920,resize_factor*1200)
    resize_factor = original_size[0]/image.shape[0]
    # Get segmentation mask
    mask = get_segmentation_mask(image,ort_session)

    write_parameters_to_file(
        inst_segmentations=mask,
        file_name=f"{output_path}/{image_name}_report.txt",
        image_name= image_name,
        image=image,
        resize_factor=resize_factor
    )

    # Visualize the results and export as an image
    overlay = visualize_instances_map(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), mask, line_thickness=3)


    # Resize all images to input image dimensions for writing
    overlay = cv2.resize(overlay,original_size[::-1], interpolation = cv2.INTER_NEAREST)
    mask = cv2.resize(mask,original_size[::-1], interpolation = cv2.INTER_NEAREST)

    cv2.imwrite(f'{output_path}/{image_name.replace(".png","")}_overlay.png',overlay)
    tifffile.imwrite(f'{output_path}/{image_name.replace(".png","")}_uint16_instance_mask.tiff',mask.astype(np.int16))
