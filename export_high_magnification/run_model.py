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

from utils import load_image_from_path, write_parameters_to_file
from inst_seg_contour import visualize_instances_map
from model import get_segmentation_mask

# download model if its not present
if not os.path.isfile("model_high_mag_maskedrcnn.onnx"):
    print('Downloading Model...')
    urllib.request.urlretrieve("https://bwsyncandshare.kit.edu/s/qLaboe98J9oZcst/download/model_high_mag_maskedrcnn.onnx", "model_high_mag_maskedrcnn.onnx")

VALID_IMAGE_FORMATS = ['.png','.tiff','.tif']

# The neural network
ort_session = onnxruntime.InferenceSession("model_high_mag_maskedrcnn.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

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
    mask_not_confident, mask_confident, mask_combined = get_segmentation_mask(image,ort_session)

    write_parameters_to_file(
        inst_segmentations=mask_not_confident,
        file_name=f"{output_path}/{image_name}_report_not_confident.txt",
        image_name= image_name,
        image=image,
        resize_factor=resize_factor
    )

    write_parameters_to_file(
        inst_segmentations=mask_confident,
        file_name=f"{output_path}/{image_name}_report_confident.txt",
        image_name= image_name,
        image=image,
        resize_factor=resize_factor
    )

    write_parameters_to_file(
        inst_segmentations=mask_combined,
        file_name=f"{output_path}/{image_name}_report_combined.txt",
        image_name= image_name,
        image=image,
        resize_factor=resize_factor
    )


    # outputs for not confident
    # Visualize the results and export as an image
    overlay = visualize_instances_map(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), mask_not_confident, line_thickness=3)
    # Resize all images to input image dimensions for writing
    overlay = cv2.resize(overlay,original_size[::-1], interpolation = cv2.INTER_NEAREST)
    mask_not_confident = cv2.resize(mask_not_confident,original_size[::-1], interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(f'{output_path}/{image_name.replace(".png","")}_overlay_not_confident.png',overlay)
    tifffile.imwrite(f'{output_path}/{image_name.replace(".png","")}_uint16_instance_mask_not_confident.tiff',mask_not_confident.astype(np.int16))


    # outputs for confident
    # Visualize the results and export as an image
    overlay = visualize_instances_map(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), mask_confident, line_thickness=3)
    # Resize all images to input image dimensions for writing
    overlay = cv2.resize(overlay,original_size[::-1], interpolation = cv2.INTER_NEAREST)
    mask_confident = cv2.resize(mask_confident,original_size[::-1], interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(f'{output_path}/{image_name.replace(".png","")}_overlay_confident.png',overlay)
    tifffile.imwrite(f'{output_path}/{image_name.replace(".png","")}_uint16_instance_mask_confident.tiff',mask_confident.astype(np.int16))


    # outputs for combined
    # Visualize the results and export as an image
    overlay = visualize_instances_map(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), mask_combined, line_thickness=3)
    # Resize all images to input image dimensions for writing
    overlay = cv2.resize(overlay,original_size[::-1], interpolation = cv2.INTER_NEAREST)
    mask_combined = cv2.resize(mask_combined,original_size[::-1], interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(f'{output_path}/{image_name.replace(".png","")}_overlay_combined.png',overlay)
    tifffile.imwrite(f'{output_path}/{image_name.replace(".png","")}_uint16_instance_mask_combined.tiff',mask_combined.astype(np.int16))

