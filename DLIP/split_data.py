import tifffile
from glob import glob



# /home/ws/kg2371/datasets/berkley_sem_unlabeled/train/samples

k=0
for img_path in glob('/home/ws/kg2371/Downloads/Phenom_Images/*'):
    img = tifffile.imread(img_path)
    for i in range(0,1024,256):
        for j in range(0,1024,256):
            print(f'{k}_{i}:{i+256} | {j}:{j+256}')
            tifffile.imwrite(f'/home/ws/kg2371/datasets/berkley_sem_unlabeled/train/samples/{k}_{i}_{j}.tif',img[i:i+256,j:j+256])
    k+=1