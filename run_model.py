import onnxruntime
import numpy as np
import cv2
import logging
import argparse
from scipy.ndimage.measurements import label
from inst_seg_contour import visualize_instances_map

def get_model_predictions(path_to_onnx, input_x):
    logging.info('Getting Model Predictions')
    # permute axes if needed
    if input_x.shape[0] != 1:
        print('Axes in wrong shape, trying to fix it by permuting.')
        input_x = np.transpose(input_x,(0,3,1,2))
    ort_session = onnxruntime.InferenceSession(path_to_onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: input_x.astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]   

def transform_to_insance_segmentations(output_y):
    logging.info('Tranforming to Instance Segmentations')
    output_y = ((output_y > 0.5)*1)[0][0]
    structure = np.ones((3,3), dtype=int)
    labeled, ncomponents = label(output_y, structure)
    instance_areas = [np.sum(labeled == x) for x in range(np.max(labeled))]
    for i in range(len(instance_areas)):
        if instance_areas[i] < MIN_OBJECT_AREA_SIZE:
            labeled[labeled == i] = 0
    return labeled



# parsing args
parser = argparse.ArgumentParser(description='Calculate Instance Segmentations to SEM Images.')
parser.add_argument(
    '--input',
    dest='input',
    type=str,
    default='./example_image.png',
    help='The input image.'
)
parser.add_argument(
    '--min_area', 
    dest='min_area',
    default=200,
    help=
        'The minimum surface area a detected object needs to have to be considered a "valid detection".\
        otherwise it will be treated as an artifact and delected. If you experience too many detections / \
        many small ones you might want to increase this value. If too few detections are the case, try to increase it'
)
args = parser.parse_args()
args = parser.parse_args()
input_image_path = args.input
MIN_OBJECT_AREA_SIZE = args.min_area

sample_img = cv2.imread(input_image_path, -1)
if sample_img is None:
    raise(Exception(f'Input image {input_image_path} not found!'))
# range [0,1] is required
sample_img = sample_img / 255
# you need to provide the input like this (batch_size,channels,height,width)
# channels needs to be = 1
# height and width need to be = 256
# we construct this shap by expanding the dims two times for channels and batch_size
sample_img = np.expand_dims(np.expand_dims(sample_img, 0),0)

# get the semantic segmentation predictions
predictions_y = get_model_predictions('model.onnx', sample_img)
cv2.imwrite(f'{input_image_path.replace(".png","")}_predictions.png',(predictions_y[0][0]>0.5)*255)
# construct instance map from semantic segmentation predictions
instances = transform_to_insance_segmentations(predictions_y)
# transform instance map to export as beatiful image
overlay = visualize_instances_map(cv2.cvtColor((sample_img[0][0]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB), instances, line_thickness=1)
cv2.imwrite(f'{input_image_path.replace(".png","")}_overlay.png',overlay)

