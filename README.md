# Careless 
Merging crystallography data without much physics. 


![Build](https://github.com/Hekstra-Lab/careless/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/Hekstra-Lab/careless/branch/main/graph/badge.svg)](https://codecov.io/gh/Hekstra-Lab/careless)
[![PyPI](https://img.shields.io/pypi/v/careless?color=blue)](https://pypi.org/project/careless/)
[![DOI](http://img.shields.io/badge/bioRxiv-10.1101/2021.01.05.425510-BD2736.svg)](https://doi.org/10.1101/2021.01.05.425510)

## Installation
```bash
pip install --upgrade pip
pip install careless
```

## Dependencies
`careless` currently supports __Python 3.7 - 3.9__. 
Pip will handle installation of all dependencies. 
`careless` uses mostly tools from the conventional scientific python stack plus
 - optimization routines from [TensorFlow](https://www.tensorflow.org/)
 - statistical distributions from [Tensorflow-Probability](https://www.tensorflow.org/probability)
 - crystallographic computing resources from 
    - [ReciprocalSpaceship](https://hekstra-lab.github.io/reciprocalspaceship/)
    - [GEMMI](https://gemmi.readthedocs.io/en/latest/)

## Examples
For `careless` usage examples and data from the [preprint](https://doi.org/10.1101/2021.01.05.425510), check out [careless-examples](https://github.com/Hekstra-Lab/careless-examples)

## Core Model

![pgm](doc/figures/pgm.svg)

`careless` uses approximate Bayesian inference to merge X-ray diffraction data. 
The model which is implemented in `careless` tries to scale individual reflection observations such that they become consistent with a set of prior beliefs.
During optimization of a model, `careless` trades off between consistency of the merged structure factor amplitudes with the data and consistency with the priors.
In essence, the optimizer tries to strike a compromise which maximizes the likelihood of the observed data while not straying far from the prior distributions. 

The implementation breaks the model down into 4 types of objects. 

### Variational Merging Model
The `VariationalMergingModel` is central object which houses the estimates of the merged structure factors.
In `careless` merged structure factors are represented by truncated normal distributions which have support on [0, ∞).
According to French and Wilson<sup>[2](#frenchwilson)</sup> this is the appropriate parameterization for acentric reflections which are by far the majority in most space groups.
These distributions are stored in the `VariationalMergingModel.surrogate_posterior` attribute. 
They serve as a parametric approximation of the true posterior which cannot easily be calculated. 
It has utility methods for training the model.
It contains an instance of each of the other objects. 
During optimization, the loss function is constructed by sampling values for the merged structure factors and scales these are combined with the prior and likelihood to compute the `Evidence Lower BOund` or (`ELBO`)
Gradiennt ascent is used to maximize the `ELBO`.


### Priors
The simplest prior which `careless` implements are the popular priors<sup>[1](#wilson)</sup> derived by A. J. C. Wilson from the random atom model. 
This is a relatively weak prior, but it is sufficient in practice for many types of crystallographic data. 
`careless` can also use reference structure amplitudes as priors. 
In this case, the structure factors are supposed to be drawn from a distribution centered at an empirical reference value. 
`careless` has reference priors implemented for Normal, Laplacian, and Student T distributions. 

### Likelihoods
The quality of the current structure factor estimates during optimization is judged by a likelihood function. 
These are symmetric probability distributions centered at the observed reflection observation. 

### Scaling Models
Right now the only model which `careless` explicitly implements is a sequential neural network model. 
This model takes reflection metadata as input and outputs a gaussian distribution of likely scale values for each reflection.
`careless` supports custom variational likelihoods as well. 
These scaling models differ from the current neural network model inasmuch as their parameters may have their own prior distributions. 

Special metadata keys for scaling. 
`careless` will happy parse any existing metadata keys in the input Mtz(s). 
During configuration some new metadata keys will be populated that are useful in many instances. 
 - <b>dHKL</b> : The inverse square of the reflection resolution. Supplying this key is a convenient way to parameterize isotropic scaling.
 - <b>file_id</b> : An integer ID unique to each input Mtz. 
 - <b>image_id</b> : An integer ID unique to each image across all input Mtzs. 
 - <b>{H,K,L}obs</b> : Internally, `careless` refers to the original miller indices from indexing as `Hobs`, `Kobs`, and `Lobs`. Supplying these three keys is the typical method to enable anisotropic scaling. 


### Considerations when choosing metadata. 
 - <b>Polarization correction</b> : Careless does not apply a specific polarization correction. 
   In order to be sure the model accounts for polarization, it is important to supply the x,y 
   coordinates of each reflection observation. 
 - <b>Isotropic scaling</b> : This is easily accounted for by supplying the 'dHKL' metadata key.
 - <b>Interleaved rotation series</b> : Most properly formatted Mtzs have a "Batch" column which contains a unique id for each image. 
   Importantly, these are usually in order. If you have time resolved data with multiple timepoints per angle, you may
   want to use the "Batch" key in conjunction with the "file_id" key. This way images from the same rotation angle will
   be constrained to scale more similarly. 
 - <b>Multi crystal scaling</b> : For scaling multiple crystals, it is best if image identifiers in the metadata do not overlap. Therefore, use the 'image_id' key. 

<a name="wilson">1</a>: Wilson, A. J. C. “The Probability Distribution of X-Ray Intensities.” Acta Crystallographica 2, no. 5 (October 2, 1949): 318–21. https://doi.org/10.1107/S0365110X49000813.

<a name="frenchwilson">2</a>: French, S., and K. Wilson. “On the Treatment of Negative Intensity Observations.” Acta Crystallographica Section A: Crystal Physics, Diffraction, Theoretical and General Crystallography 34, no. 4 (July 1, 1978): 517–25. https://doi.org/10.1107/S0567739478001114.

