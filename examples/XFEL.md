### Merging Serial XFEL Data with Careless

The data in `careless/examples/thermolysin_xfel` are serial crystallography data from thermolysin microcrystals. 
The parent experiment of these data is freely available at the [CXIDB](https://cxidb.org/id-81.html). 
Because the original data set is very large, this example deals with a single run containing 3,160 images.
The unmerged reflections, stored in `careless/examples/thermolysin_xfel/unmerged.mtz` were prepared with a custom `cctbx` [script](../scripts/stills2mtz). 


Enter the thermolysin directory and use [reciprocalspaceship](https://github.com/hekstra-lab/reciprocalspaceship) to explore the contents of this mtz file. 

```bash
cd careless/examples/thermolysin_xfel
rs.mtzdump unmerged.mtz
```
The output will look like this:
```
Spacegroup: P6122
Extended Hermann-Mauguin name: P 61 2 2
Unit cell dimensions: 93.239 93.239 130.707 90.000 90.000 120.000

mtz.head():

           BATCH  cartesian_fixed_obs_x  ...  cartesian_delta_z  PARTIAL
H   K  L                                 ...                            
-37 -4 18      0            -0.39638376  ...     -6.3847256e-06    False
-35 -4 25      0            -0.37496334  ...      -9.921834e-06    False
    -3 19      0            -0.37502986  ...      1.8138476e-07    False
    -2 2       0            -0.37495732  ...      4.3450336e-06    False
       3       0            -0.37503228  ...     -2.2314734e-07    False

[5 rows x 20 columns]

mtz.describe():

           BATCH  cartesian_fixed_obs_x  ...  cartesian_delta_y  cartesian_delta_z
count  1.098e+06              1.098e+06  ...          1.098e+06          1.098e+06
mean   1.617e+03             -7.497e-04  ...          1.626e-07         -8.234e-08
std    9.149e+02              1.993e-01  ...          1.490e-04          1.495e-04
min    0.000e+00             -5.810e-01  ...         -1.727e-03         -1.991e-03
25%    8.060e+02             -1.391e-01  ...         -5.915e-05         -3.389e-05
50%    1.664e+03             -1.192e-05  ...          6.831e-08          2.921e-08
75%    2.404e+03              1.290e-01  ...          5.959e-05          3.388e-05
max    3.159e+03              5.905e-01  ...          1.794e-03          2.004e-03

[8 rows x 19 columns]

mtz.dtypes:

BATCH                        Batch
cartesian_fixed_obs_x      MTZReal
cartesian_fixed_obs_y      MTZReal
cartesian_fixed_obs_z      MTZReal
cartesian_fixed_x          MTZReal
cartesian_fixed_y          MTZReal
cartesian_fixed_z          MTZReal
ewald_offset               MTZReal
I                        Intensity
SigI                        Stddev
xcal                       MTZReal
ycal                       MTZReal
xobs                       MTZReal
yobs                       MTZReal
sigxobs                    MTZReal
sigyobs                    MTZReal
cartesian_delta_x          MTZReal
cartesian_delta_y          MTZReal
cartesian_delta_z          MTZReal
PARTIAL                       bool
dtype: object
```

There are many columns from the [stills2mtz](../scripts/stills2mtz) script. 
For merging XFEL data we are most interested in the `cartesian_delta_{x,y,z}` columns. 
These contain the 3-dimensional ewald offset vector between the observed reflection centroids and their centroids in reciprocal space. 
These are in a crystal-fixed cartesian [coordinate system](https://dials.github.io/documentation/conventions.html). 
Because we're dealing with still images here, each of the reflections have partial intensities which are dictated by how far away from the ideal Bragg contition they fall. 
The 3D Ewald offset vector is a way of summarizing how disastified Bragg's law is for a particular reflection observation.
We will achieve the best merging performance with `Careless` if we include these vectors in the metadata supplied to the scaling model. 

