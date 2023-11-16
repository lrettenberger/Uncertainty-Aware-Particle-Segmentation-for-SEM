Run

`conda env create -f environment.yml`

To create the conda environment.
- Execute `python run_model.py` to evaluate the images within the `images` directory.
- You can change the input dir with the `--input` parameter (e.g. `python run_model.py --input /some/other/dir`).
- You can also provide a single file: `python run_model.py --input /some/other/dir/image.png`
- Only `.tiff`, `.tif`, and `.png` inputs are supported.
- Images should be provided as `uint8` files in the range `[0,255]`. 
- With the default installation this will run on the CPU, so it will probably be rather slow. If you have a GPU, you can install onnxruntime-gpu via pip to have better performance!
- This implementation expects images to be in the size `(1920,1200)`. Any other size is resized to be `(1920,1200)`. If the image size is not a multiple of `(1920,1200)` the results will most likely be bad. 
- The script will create a `out/` directory in which the segmentation masks are stored
- One file with the original file name and the suffix `_uint16_instance_mask_confident.tiff` will be generated that contains the particles found with high confidence, one file with the suffix `_uint16_instance_mask_not_confident.tiff` containing the particles with low confidence, and one file with the suffix `_uint16_instance_mask_combined.tiff` with all found particles (no distinction).
- The detected particles are uint16 integer encoded.
