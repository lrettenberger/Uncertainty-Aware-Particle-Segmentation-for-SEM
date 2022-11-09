# Scripts
This module contains the Python files to execute certain experiments. `train.py` executes experiments, defined by an yaml file. `train.py` expects two parameters: `--config_files` which defines the path to the config file(s) and `--result_dir` which indicates, where the resulting files should be saved (e.g. training weights).

Example call to `train.py`: 

`python3 train.py --config_files ./DLIP/experiments/configurations/example/general_configuration.yaml ./DLIP/experiments/configurations/example/example_configuration.yaml --result_dir ~/results_example_experiment`

## Structure
```
.
├── base_classes # Base classes, which should be extended to define new datasets.
│   └── segmentation # Base classes, specifically for segmentation datasets.
├── example #  An example dataset, to obtain an overview how new datasets should be defined.
```
