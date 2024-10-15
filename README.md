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

## Dataset Attributes:

* Dataset size: the maximum number of generated images.
* Sampling strategy: define the technique for sampling a probability distribution (Monte Carlo, Latin HyperCube Sampling, or Low Discrepancy Sequence).
* Random seed: the number used to initialize the random generator.
* Resolution: the resolution in pixels of the image's sides.
* Background color: The color in the hexadecimal value of the image's background.
* Allow out-of-border: if you select this option, shapes can extend beyond the image borders.

## Latent factor values:

* Color: all colors can be chosen. Colors can be selected using the color wheel or by entering the hex code directly.
* Shape: 2D regular shapes. Shapes are identified by the number of vertices (=: circle, 3:triangle, 4:square, and so on)
* Scale: 6 values linearly spaced in [0.5, 1]
* Orientation: 40 values in [0, 2 pi]
* Position X: 32 values in [0, 1]
* Position Y: 32 values in [0, 1]
  
imgs: (737280 x 64 x 64, uint8) Images in black and white.
latents_values: (737280 x 6, float64) Values of the latent factors.
latents_classes: (737280 x 6, int64) Integer index of the latent factor values. Useful as classification targets.
metadata: some additional information, including the possible latent values.
Alternatively, a HDF5 version is also available, containing the same data, packed as Groups and Datasets.

##Multivariate distribution:
Each of the latent variables and the uncertainty measures can be described by one of the following distributions
* Constant (fix the value of a latent variable to simplify the problem)
* Uniform distribution (equivalent to generating all possibilities)
* Gaussian distribution
* Truncated Gaussian
In addition, each of the above distributions is associated with a correlation matrix, which makes it possible to generate a corresponding multivariate distribution according to Sklar's theorem.  Thus, the latent variables can be generated to be dependent on each other.


## Disentanglement metrics:
This dataset was created as a unit test of disentanglement properties of unsupervised models under different probability distributions of the latent variables. NA_DA can be used to determine how well models recover the ground truth latents presented above, especially when their distributions are altered compared to the standard uniform ones proposed in most synthetic datasets.

## Uncertainty Quantification and Reliability:
This database generator is suitable for uncertainty quantification and reliability analysis on synthetic images (i.e., to test models developed using Bayesian formalism). In particular, different levels of uncertainty (both Epistemic and Aleatoric) can be imposed on the dataset to assess the ability of the model to produce correct reliability estimates.


