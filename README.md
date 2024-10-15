# NA_DA: Not-A-DAtabase generator of probabilistic synthetic data 

![anim](https://github.com/user-attachments/assets/22cff88b-ac69-4542-81b3-13318474b0a3)

This repository contains the dSprites dataset, used to assess the disentanglement properties of unsupervised learning methods.

If you use this dataset in your work, please cite it as follows:

Bibtex
@misc{dsprites17,
author = {Loic Matthey and Irina Higgins and Demis Hassabis and Alexander Lerchner},
title = {dSprites: Disentanglement testing Sprites dataset},
howpublished= {https://github.com/deepmind/dsprites-dataset/},
year = "2017",
}
Description
dsprite_gif

dSprites is a dataset of 2D shapes procedurally generated from 6 ground truth independent latent factors. These factors are color, shape, scale, rotation, x and y positions of a sprite.

All possible combinations of these latents are present exactly once, generating N = 737280 total images.

## Latent factor values
Color: white
Shape: square, ellipse, heart
Scale: 6 values linearly spaced in [0.5, 1]
Orientation: 40 values in [0, 2 pi]
Position X: 32 values in [0, 1]
Position Y: 32 values in [0, 1]
We varied one latent at a time (starting from Position Y, then Position X, etc), and sequentially stored the images in fixed order. Hence the order along the first dimension is fixed and allows you to map back to the value of the latents corresponding to that image.

We chose the latents values deliberately to have the smallest step changes while ensuring that all pixel outputs were different. No noise was added.

The data is a NPZ NumPy archive with the following fields:

imgs: (737280 x 64 x 64, uint8) Images in black and white.
latents_values: (737280 x 6, float64) Values of the latent factors.
latents_classes: (737280 x 6, int64) Integer index of the latent factor values. Useful as classification targets.
metadata: some additional information, including the possible latent values.
Alternatively, a HDF5 version is also available, containing the same data, packed as Groups and Datasets.

## Disentanglement metric
This dataset was created as a unit test of disentanglement properties of unsupervised models. It can be used to determine how well models recover the ground truth latents presented above.

## Uncertainty measures

