# Physics-Based Spectral Scaling

For Laue diffraction experiments where the incident spectrum is well-characterized, `careless` allows you to bypass the neural network scaler and strictly enforce a physical scaling model. This is particularly useful for resolving the ambiguity between harmonic reflections ($n=1, 2, \dots$) when the dataset does not have sufficient wavelength redundancy to learn the spectral shape from scratch.

## Usage

To enable this mode, provide a two-column text file containing the incident spectrum using the `--spectral-file` argument. Note that a dummy metadata key (`WAVEL`) is still required, but it is unused.

```bash
careless poly \
  "reflection_file.mtz" \
  "output_root" \
  --spectral-file source_spectrum.txt \
  --lorentz-correction \
  --kl-weight 0.5 \
  --student-t-likelihood-dof 4 \
  "WAVEL"
```

## Spectrum File Format

The spectral file should be a whitespace-separated text file with two columns:

1.  **Wavelength ($\mathring{A}$)**
2.  **Scale / Flux (Arbitrary Units)**

<!-- end list -->

```text
0.95  0.1
1.00  1.0
1.05  0.9
...
```

The model uses linear interpolation to determine the scale factor for any given reflection wavelength. A lookup table is pre-calculated for performance, which can be tuned via `--spectral-grid-points`.

## Lorentz Correction

The Laue Lorentz factor correction accounts for the geometric probability of diffraction as a function of scattering angle $\theta$ and wavelength $\lambda$.

$$ L \propto \frac{\lambda^4}{\sin^2 \theta} = 4 \lambda^2 d^2 $$

When `--lorentz-correction` is enabled, the interpolated spectral scale $S(\lambda)$ is multiplied by this factor:

$$ \text{Scale}_{total} = S(\lambda) \times 4 \lambda^2 d^2 $$

<a name="lange">1</a>: Lange, J. "The Lorentz Factor for the Laue Technique." [Acta Crystallographica Section A 51, no. 4 (1995): 559–565.](https://doi.org/10.1107/S0108767395001358)


## Trainable Scale

By default, the spectral curve is fixed. To allow the global intensity to float (matching the magnitude of the data) while preserving the spectral shape, add the `--trainable-spectral-scale` flag. This introduces a single learnable scalar multiplier $A$:

$$ \text{Scale}_{total} = A \times S(\lambda) \times L_{correction} $$
