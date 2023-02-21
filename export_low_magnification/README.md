Run

`conda env create -f environment.yml`

To create the conda environment.
- Execute `python run_model.py` to evaluate the images within the `images` directory.
- You can change the input dir with the `--input` parameter (e.g. `python run_model.py --input /some/other/dir`).
- You can also provide a single file: `python run_model.py --input /some/other/dir/image.png`
- Only `.tiff`, `.tif`, and `.png` inputs are supported.
- Images should be provided as `uint8` files in the range `[0,255]`. 
- An overlay image of the predicted particles and a `.txt` file with estimated parameters will be created for each image.
- With the default installation this will run on the CPU, so it will probably be rather slow. If you have a GPU, you can install onnxruntime-gpu via pip to have better performance!
- This implementation is invariant to the image size since it uses patching with a patch size of `1024x1024`. Small images are padded to a minium size of `1024x1024` (which is not visible to the user, since the padding is removed before output generation). Bigger images will slow down processing but should not make the program crash.