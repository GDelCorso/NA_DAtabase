# NA_DA: Not-A-DAtabase generator of probabilistic synthetic data 

![anim](https://github.com/user-attachments/assets/22cff88b-ac69-4542-81b3-13318474b0a3)

NA_DA is an open-source software written in Python that generates datasets of regular two-dimensional geometric shapes based on probabilistic distributions.
NA_DA comes with an intuitive GUI (Graphical User Interface) that allows users to define shapes, colors, and distributions of features of datasets consisting of image sets and CSV files containing metadata for each element. These databases can be saved to provide a unique identifier of the dataset, allowing perfect reproducibility or easy modification of the dataset using the GUI or directly by calling the generator class. Therefore, NA_DA is a tool to help and support the investigation of trustworthiness, overconfidence, uncertainty, and computation time of machine learning and deep learning models.
 
If you use this dataset in your work, please cite it as follows:

## Bibtex

```
@article{volpinidatabase,
  title={NA DAtabase: Generator of Probabilistic Synthetic Geometrical Shape Dataset},
  author={Volpini, Federico and Caudai, Claudia and Del Corso, Giulio and Colantonio, Sara}
}
```

NA_DA can generate datasets of 2D shapes with customizable dataset attributes, latent variables, uncertainty and deformation and their multivariate distributions.

## Dataset Attributes

* Dataset size:
* Sampling strategy: 
* Random seed:
* Resolution:
* Background color:
* Allow out of border:

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
This dataset was created as a unit test of disentanglement properties of unsupervised models under different probability distributions of the latent variables. NA_DA can be used to determine how well models recover the ground truth latents presented above, especially when their distributions are altered compared to the standard uniform ones proposed in most synthetic datasets.

## Uncertainty Quantification and Reliability:
This database generator is suitable for uncertainty quantification and reliability analysis on synthetic images (i.e., to test models developed using Bayesian formalism). In particular, different levels of uncertainty (both Epistemic and Aleatoric) can be imposed on the dataset to assess the ability of the model to produce correct reliability estimates.


