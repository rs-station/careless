### Careless Zero™

The implementation of variational scaling inside of Careless is a tad convoluted, because it is designed for flexibility. 
For those users that want to know more about how Careless actually works, but don't care to sift through the actual source, I've provided this super minimalist [implementation](https://github.com/Hekstra-Lab/careless/blob/master/examples/careless_zero/careless_zero.py) in 100 lines of python. 

The script merges X-ray data using [reciprocalspaceship](https://github.com/hekstra-lab/reciprocalspaceship) for data i/o and [tensorflow_probability](https://www.tensorflow.org/probability) for probabilistic programming.
`careless_zero.py` inplements the default Careless model and uses it to merge the lysozyme SAD [data set](https://github.com/Hekstra-Lab/careless/blob/master/examples/HEWLSSAD.md).
To run the script, first enter the careless zero directory,

```bash
cd {path to installation}/careless/examples/careless_zero
```
and type

```bash
python careless_zero.py
```

The script will take several minutes to run on a very fast GPU and tens of minutes on a cpu. Once finished, you can use the refinement script in the `careless_zero` directory to make a map.

```bash
phenix.refine refine.eff
```

Inspect the map using `coot`
```bash
coot hewl_refine_1.pdb hewl_refine_1.mtz
```
