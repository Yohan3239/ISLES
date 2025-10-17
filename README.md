A fully custom 3D Convolutional Neural Network for ischemic stroke lesion segmentation from MRI data, implemented entirely in C++ with CUDA acceleration — no external ML frameworks.
Developed for the ISLES 2022 dataset using FLAIR, DWI, and ADC modalities.

**Features**

Custom-built CNN architecture with convolution, pooling, upsampling, and trilinear interpolation layers

CUDA-accelerated forward and backward propagation

Composite loss function: Tversky + Focal for class imbalance handling

LeakyReLU and sigmoid activations implemented from scratch

Affine transformations for multi-modality MRI registration

3D volumetric output reconstruction with voxel-level accuracy metrics

[![DEMO:][(https://img.youtube.com/vi/<VIDEO_ID>/0.jpg)](https://youtu.be/<VIDEO_ID>](https://youtu.be/II0lLBmuz-k))

**Dataset**

ISLES 2022 Dataset: 250 MRI cases (FLAIR, DWI, ADC + ground truth masks)

Data normalised, registered via affine transformations, and augmented (rotation, scaling, intensity shifts).
(Dataset not included – available at https://zenodo.org/records/7153326)

MIT License

Copyright (c) 2025 Yohan Lee

