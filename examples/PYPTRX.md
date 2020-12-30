### PYP Time Resolved Data
The goal of this example is to demonstrate how `Careless` can be used to create an isomorphous difference map between time points in a time resolved experiment. 
You will find data from a time resolved experiment conducted at BioCARS on photoactive yellow protein (PYP) in `careless/examples/data/pyp`. 
PYP undergoes a trans to cis isomerization when it is exposed to blue light. 
The data set consisted of 20 dark images and 20 images which were acquired 2ms after the arrival of a blue laser pulse. 
For starters, let's enter the data directory and have a look around. 
Type 
```bash
cd [Path to your careless installation]/examples/pyp
ls -R
```

You will see the following items:
 - 2ms.mtz (Unmerged reflections from the 2ms timepoint)
 - off.mtz (Unmerged reflections from the dark images)
 - refine.eff (A PHENIX refinement parameter file)
 - reference_data/ligands.cif (A parameter file which tells PHENIX how to interpret the chromophore during refinement)
 - reference_data/2ms.pdb (Atomic model for the excited state)
 - reference_data/off.pdb (Atomic model of the ground state)

Because we will have to supply metadata from the mtz files, let's first have  look at what is inside them.
This can be easily done with a command line script supplied by [ReciprocalSpaceship](https://hekstra-lab.github.io/reciprocalspaceship/) which will have been installed when you installed `Careless`. Type `rs.mtzdump off.mtz`; you should something like the following output. 

```
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
```

From this we can see what metadata we have in the files. 
We will choose to use the following metadata in our `careless` model
 - `X` (The detector X position in pixels for the Bragg peak)
 - `Y` (The detector Y position in pixels for the Bragg peak)
 - `Wavelength` (The wavelength of each reflection)
 - `BATCH` (The image number on which each reflection was observed)
 - `dHKL` (The resolution of the reflection) 
 - `Hobs,Kobs,Lobs` (The observed, P1 miller indices of the reflections). 
 
dHKL, Hobs, Kobs, and Lobs are all special metadata keys which are always made avaible to `careless` models. The combination `dHKL,Hobs,Kobs,Lobs` implies anisotropic scaling. 
Now that we have identified the metadata keys we want to use, we can create an output directory and run `careless`.

```bash
mkdir merge
careless poly \
  --separate-files \
  --iterations=10000 \
  --wavelength-key='Wavelength' \
  "X,Y,Wavelength,BATCH,dHKL,Hobs,Kobs,Lobs" \
  off.mtz \
  2ms.mtz \
  merge/pyp
```

Here's a breakdown of what each argument means.
- the `--separate-files` flag tells `careless` we would like to keep the reflection sin `off.mtz` and `2ms.mtz` separate during merging. Without this flag, `careless` would output a single `mtz` file containing the average structure factors for the two data sets. 
- `--iterations` is how many gradient steps to take 
- when processing polychromatic data, it is necessary to provide a `--wavelength-key`. `careless` will use this information to structure the harmonic deconvolution in the likelihood function. 
- Immediately after the optional `--` arguments, the user must supply a *comma separated* string of *metadata keys*.
- the next block of arguments are the *input mtz(s)*
- the final argument is always the *output filename* base.

Running the optimization will take different amounts of time depending on your particular hardware. 
On a powerful CPU, it will likely take about 10 minutes. 
However, with a relatively recent NVIDIA GPU it will take just a few minutes. 
Once it is completed, the output files will appear in the `merge/` directory. 
The output will begin with the base filename supplied as the last argument to careless. 
There will be three files for each input mtz. 

- pyp_0.mtz - merged data from the first mtz (off.mtz)
- pyp_1.mtz - merged data from the second mtz (2ms.mtz)
- pyp_half1_0.mtz - merged data from the first mtz and first half data set
- pyp_half1_1.mtz - merged data from the second mtz and first half data set
- pyp_half2_0.mtz - merged data from the first mtz and second half data set
- pyp_half2_1.mtz - merged data from the second mtz and second half data set
- pyp_losses.npy  - loss function values for the full data set
- pyp_half1_losses.mtz - loss function values for the first half data set
- pyp_half2_losses.mtz - loss function values for the second half data set

To make a difference map from these data, we first need to refine the dark data. 

```bash
mkdir phenix
phenix.refine refine.eff
```

We can now use coot to have a look at the electron density map by calling 

```bash
coot phenix/PYP_refine_1.mtz \
    reference_data/off.pdb \
    reference_data/2ms.pdb 
```


You can quickly find the chromophore by pressing `ctrl-l`.

![2fo-fc map](images/pyp-2fo-fc.gif)

Making a difference map using `make_difference_map`. Run the following

```bash
make_difference_map merge/pyp_0.mtz merge/pyp_1.mtz phenix/PYP_refine_1.mtz
```

which will generate `difference.mtz`. 

![load-differences](images/pyp-load-differences.gif)

