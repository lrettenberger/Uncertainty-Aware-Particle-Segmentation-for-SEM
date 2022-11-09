# Models
This module contains the model definitions. The models are defined in a flexible, resuable way. The `zoo/` directory contains the specific models. In `zoo/building_blocks` atomic functionalities are defined. `zoo/encode` defines components to encode images (i.e. feature vectors), `zoo/decode` to decode feature vectors. `zoo/compositions` contains modules usable for training with Pytorch Lightning, which are compositions of encoder, decoders and/or building blocks.
## Structure
```
.
└── zoo
    ├── building_blocks
    ├── compositions
    ├── decoder
    └── encoder
```
