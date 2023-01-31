import tifffile
import cv2
import numpy as np

from parameter_estimation import get_particle_sizes

def load_image_from_path(input_image_path):
    # Load the image
    if input_image_path.split('.')[-1] == 'tiff' or input_image_path.split('.')[-1] == 'tif':
        image = tifffile.imread(input_image_path)
    else:
        image = cv2.imread(input_image_path,-1)
    if image is None:
        raise(Exception(f'Input image {input_image_path} not found!'))
    
    if image.dtype == np.uint32 or image.dtype == np.uint16:
        raise Exception('uint32 and uint16 not supported, cant process image {input_image_path}')
    if np.max(image) > 255:
        raise Exception(f'Image Range not correct. Expect the range to be [0,255] but is actually bigger [{np.min(image)},{np.max(image)}]')
    
    if image.dtype == np.float32:
        # we need uint8 ranges
        if np.max(image) <= 1:
            # range [0,1] is assumed, so multiply by 255 to get [0,255]
            image = (image*255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    if len(image.shape) > 2:
        if image.shape[2] > 1:
            print('More than one channel detected. Using just the first one. As this program expects single channel images you should look into this.')
        image = image[:,:,0]
    return image

def write_parameters_to_file(inst_segmentations,file_name,image_name,image):
    # Calculate the sizes of the detected particles as pixel sizes
    # Writing it to a report file
    sizes_by_pixels = get_particle_sizes(inst_segmentations)

    # Write small report about the sizes
    f = open(file_name, "w")
    f.write(f'Report for {image_name} \n')
    f.write('----------------------------------- \n')
    image_size = np.prod(image.shape)
    f.write(f'Particle density: {(np.sum(sizes_by_pixels)/image_size)*100:.2f}% \n')
    f.write('Individual particle sizes (absolute | relative) \n')
    for particle_size in sizes_by_pixels:
        f.write(f'({particle_size} | {(particle_size/image_size)*100:.2f}%) \n')
    f.close()