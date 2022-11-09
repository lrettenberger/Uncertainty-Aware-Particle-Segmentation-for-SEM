# Data
This module contains the Dataset (and Datamodule) definitions. The example module shows an exemplary image dataset. To be usable in experiments the datasets need to be explicitly defined in the `__init__.py` file.
## Structure
```
.
├── base_classes # Base classes, which should be extended to define new datasets.
│   └── segmentation # Base classes, specifically for segmentation datasets.
├── example #  An example dataset, to obtain an overview how new datasets should be defined.
```
