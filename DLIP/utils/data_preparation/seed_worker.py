import torch
import numpy as np
import random
import imgaug

"""
i) Using PyTorch + NumPy? You're making a mistake.
A bug that plagues thousands of open-source ML projects.
https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
ii) Albumentations
https://github.com/albumentations-team/albumentations/issues/873
iii) A way to handle this issuue(soure
https://pytorch.org/docs/stable/notes/randomness.html)
DataLoader will reseed workers following Randomness in multi-process
data loading algorithm. Use worker_init_fn() to preserve
reproducibility:
"""

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    imgaug.seed(worker_seed)