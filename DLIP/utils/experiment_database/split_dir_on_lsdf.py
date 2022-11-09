import os

def split_dir_if_on_lsdf(dir):
    dir_split = dir.split(os.path.sep)
    if 'iai-aida' in dir_split:
        return (os.path.sep).join(dir_split[dir_split.index('iai-aida'):])
    return (os.path.sep).join(dir_split)
