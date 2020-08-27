# Careless 
Merging crystallography data without much physics. 

![Build](https://github.com/Hekstra-Lab/careless/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/Hekstra-Lab/careless/branch/master/graph/badge.svg?token=Y39W8T060L)](https://codecov.io/gh/Hekstra-Lab/careless)


## Installation
    pip install -U pip
    git clone git@github.com:Hekstra-Lab/careless
    cd careless
    pip -e install .

## Dependencies
Pip will handle installation of all dependencies. 
`careless` uses mostly tools from the conventional scientific python stack plus
 - optimization routines from [TensorFlow](https://www.tensorflow.org/)
 - statistical distributions from [Tensorflow-Probability](https://www.tensorflow.org/probability)
 - crystallographic computing resources from 
    - [ReciprocalSpaceship](https://hekstra-lab.github.io/reciprocalspaceship/)
    - [GEMMI](https://gemmi.readthedocs.io/en/latest/)


## Core Model
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
 - <b>Z-{intensity key,sigma key}</b> : Careless supplies a Z-score of the intensity and sigma intensity for each reflection observation calculated amongst the set of redundant symmetry equivalents being merged. These inherit the name of the intensity/sigma intensity keys being used for merging prefixed by `'Z-'`. One might consider supplying these equivalent to using a sort of weighted likelihood function. 


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

## Examples

### PYP Time Resolved Data
The goal of this example is to demonstrate how `Careless` can be used to create an isomorphous difference map between time points in a time resolved experiment. 
You will find data from a time resolved experiment conducted at BioCARS on photoactive yellow protein (PYP). 
PYP undergoes a trans to cis isomerization when it is exposed to blue light. 
The data set consisted of 20 dark images and 20 images which were acquired 2ms after the arrival of a blue laser pulse. 
For starters, let's enter the data directory and have a look around. 
Type 
    cd [Path to your careless installation]/data/laue/pyp
    ls

You will see the following items:
 - 2ms.pdb (Atomic model for the excited state)
 - 2ms_varEll.mtz (Unmerged reflections from the 2ms timepoint)
 - ligands.cif (A parameter file which tells PHENIX how to interpret the chromophore during refinement)
 - off.pdb (Atomic model of the ground state)
 - off_varEll.mtz (Unmerged reflections from the dark images)
 - refine.eff (A PHENIX refinement parameter file)

Because we will have to supply metadata from the mtz files, let's first have  look at what is inside them.
This can be easily done with a command line script supplied by [ReciprocalSpaceship](https://hekstra-lab.github.io/reciprocalspaceship/) which will have been installed when you installed `Careless`. Type `rs.mtzdump off_varEll.mtz`; you should something like the following output. 
    Spacegroup: P63
    Extended Hermann-Mauguin name: P 63
    Unit cell dimensions: 66.900 66.900 40.954 90.000 90.000 120.000
    
    mtz.head():
    
                  X      Y  Wavelength     I  SigI  BATCH  PARTIAL
    H  K   L
    14 -10 -24 12.4  992.8   1.0406566 237.5  38.6      0    False
    13 -7  -22 20.3 1063.3   1.1311072 46.25 35.82      0    False
    14 -5  -23 20.7 1139.3   1.0808493  95.0 32.92      0    False
       -9  -24 21.0 1021.5   1.0358433  98.0 32.48      0    False
    13 -6  -22 34.7 1093.3   1.1206307 28.75 27.95      0    False
    
    mtz.describe():
    
                   X          Y  Wavelength          I       SigI      BATCH
    count  44914.000  44914.000   44914.000  4.491e+04  44914.000  44914.000
    mean    1021.107   1019.350       1.091  4.790e+03     79.440      9.485
    std      537.324    540.376       0.043  3.003e+04     75.860      5.764
    min       12.400     10.900       1.020  1.050e+00     16.880      0.000
    25%      545.800    541.600       1.054  1.095e+02     49.360      4.000
    50%     1038.000   1015.850       1.086  3.442e+02     58.270      9.000
    75%     1491.075   1495.175       1.124  1.410e+03     78.758     14.000
    max     2037.600   2037.100       1.180  1.028e+06   1594.960     19.000
    
    mtz.dtypes:
    
    X               MTZReal
    Y               MTZReal
    Wavelength      MTZReal
    I             Intensity
    SigI             Stddev
    BATCH             Batch
    PARTIAL            bool
    dtype: object

From this we can see what metadata we have in the files. 
We will choose to use the following metadata in our `careless` model
 - X (The detector X position in pixels for the Bragg peak)
 - Y (The detector Y position in pixels for the Bragg peak)
 - Wavelength (The wavelength of each reflection)
 - BATCH (The image number on which each reflection was observed)
 - dHKL (The resolution of the reflection. This is a special metadata key which is always made available in `careless` models. You can read more about special keys `Scaling Models` section above)

Now that we have identified the metadata keys we want to use, we can create an output directory and run `careless`.

	mkir merge
    careless poly \
      --separate-files \
      --iterations=30000 \
      --learning-rate=0.001 \
      --isigi-cutoff=1. \
      --wavelength-key='Wavelength' \
      "X,Y,Wavelength,BATCH,dHKL" \
      off_varEll.mtz \
      2ms_varEll.mtz \
      merge/pyp

Here's a breakdown of what each argument means.
 - the `--separate-files` flag tells `careless` we would like to keep the reflection sin `off_varEll.mtz` and `2ms_varEll.mtz` separate during merging. Without this flag, `careless` would output a single `mtz` file containing the average structure factors for the two data sets. 
 - `--iterations` is how many gradient steps to take 
 - `--learning-rate` is the learning rate used by the Adam optimizer, and 0.001 is the default value.
 - the `--isigi-cutoff` tells `careless` to discard reflections for which `I/Sigi < 1.`. Without this flag, `careless` would produce a more complete data set, but the difference signal would be somewhat diminished. 
 - when processing polychromatic data, it is necessary to provide a `--wavelength-key`. `careless` will use this information to structure the harmonic deconvolution in the likelihood function. 
 - Immediately after the optional `--` arguments, the user must supply a *comma separated* string of *metadata keys*.
 - the next block of arguments are the *input mtz(s)*
 - the final argument is always the *output filename* base.

Running the optimization will take different amounts of time depending on your particular hardware. 
On a powerful CPU, it will likely take 30-45 minutes.
However, with a relatively recent NVIDIA GPU it will take just a few minutes. 
Once it is completed, the output files will appear in the `merge/` directory. 
The output will begin with the base filename supplied as the last argument to careless. 
There will be three files for each input mtz. 

 - pyp_0.mtz - merged data from the first mtz (off_varEll.mtz)
 - pyp_1.mtz - merged data from the second mtz (2ms_varEll.mtz)
 - pyp_half1_0.mtz - merged data from the first mtz and first half data set
 - pyp_half1_1.mtz - merged data from the second mtz and first half data set
 - pyp_half2_0.mtz - merged data from the first mtz and second half data set
 - pyp_half2_1.mtz - merged data from the second mtz and second half data set
 - pyp_losses.npy  - loss function values for the full data set
 - pyp_half1_losses.mtz - loss function balues for the first half data set
 - pyp_half2_losses.mtz - loss function balues for the second half data set

To make a difference map from these data, we first need to refine the dark data. 

    cd merge
    mkdir phenix
    cd phenix
    phenix.refine ../pyp_0.mtz ../../off.pdb ../../ligands.cif ../../refine.eff

We can now use coot to have a look at the electron density map by calling `coot PYP_refine_1.mtz PYP_refine_1.pdb`
You can quickly find the chromophore by pressing `ctrl-l`.

![2fo-fc map](data/images/pyp-2fo-fc.apng)
