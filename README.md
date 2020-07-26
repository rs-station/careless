### Careless
![Build](https://github.com/Hekstra-Lab/careless/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/Hekstra-Lab/careless/branch/master/graph/badge.svg?token=Y39W8T060L)](https://codecov.io/gh/Hekstra-Lab/careless)



Merging crystallography data without much physics. 


# Core Model
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
They serve as a parametric approximation of the ture posterior which cannot easily be calculated. 
It has utility methods for training the model.
It contains an instance of each of the other objects. 
During optimization, the loss function is constructed by sampling values for the merged structure factors and scales these are combined with the prior and likelihood to compute the `Evidence Lower BOund` or (`ELBO`)
Gradiennt ascent is used to maximize the `ELBO`.


### Priors
The simplest prior which `careless` implements are the popular priors<sup>[1](#wilson)</sup> derived by A. J. C. Wilson from the random atom model. 
This is a relatively weak prior, but it is sufficient in practice for many types of crystallographic data. 
`careless` can also use reference structure amplitudes as priors. 
In this case, the structure factors are supposed to be drawn from a distributrion centered at an empirical reference value. 
`careless` has reference priors implemented for Normal, Laplacian, and Student T distributions. 

### Likelihoods
The quality of the current structure factor estimates during optimization is judged by a likelihood function. 
These are symmetric probability distributions centered at the observed reflection observation. 

### Scaling Models
Right now the only model which `careless` explicitly implements is a sequential neural network model. 
This model takes reflection metadata as input and outputs a gaussian distribution of likely scale values for each reflection.
`careless` supports custom variational likelihoods as well. 
These scaling models differ from the current neural network model inasmuch as their parameters may have their own prior distributions. 


<a name="wilson">1</a>: Wilson, A. J. C. “The Probability Distribution of X-Ray Intensities.” Acta Crystallographica 2, no. 5 (October 2, 1949): 318–21. https://doi.org/10.1107/S0365110X49000813.

<a name="frenchwilson">2</a>: French, S., and K. Wilson. “On the Treatment of Negative Intensity Observations.” Acta Crystallographica Section A: Crystal Physics, Diffraction, Theoretical and General Crystallography 34, no. 4 (July 1, 1978): 517–25. https://doi.org/10.1107/S0567739478001114.


