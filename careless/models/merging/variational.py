import torch
import torch.nn as nn
import numpy as np
import lightning as L
from careless.models.base import (
    BaseModel,
    reset_losses_and_metrics,
    get_accumulated_losses,
    get_accumulated_metrics,
)
from careless.distributions import TruncatedNormal
from careless.optim import AdamEpsInsideSqrt


class VariationalMergingModel(L.LightningModule, BaseModel):
    """
    Central variational merging model.

    Maximises the ELBO:
        ELBO = E_q[log p(I | F, Σ)] - KL[q(F) || p(F)]

    Uses PyTorch Lightning for training orchestration and mirrors the Keras
    add_loss / add_metric API via thread-local loss accumulation in BaseModel.
    """

    def __init__(
        self,
        surrogate_posterior,
        prior,
        likelihood,
        scaling_model,
        mc_sample_size=1,
        kl_weight=None,
        scale_kl_weight=None,
        scale_prior=None,
        # optimizer hyperparameters
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        adam_epsilon=1e-7,
        filter_nan_gradients=True,
    ):
        """
        Parameters
        ----------
        surrogate_posterior : TruncatedNormal or similar nn.Module
            Learnable distribution over structure factor amplitudes q(F).
        prior : Prior
            Prior distribution p(F); must implement log_prob(F).
        likelihood : Likelihood
            Observation likelihood p(I | F, Σ).
        scaling_model : Scaler
            Maps reflection metadata → scale distribution q(Σ).
        mc_sample_size : int
            Number of MC samples for ELBO estimation.
        kl_weight : float or None
            If None, KL is summed (divided by mc_sample_size); otherwise it weights
            a per-sample mean KL.
        scale_kl_weight : float or None
            Same as kl_weight but for the scale KL term.
        scale_prior : distribution or None
            Optional prior on scale factors.
        learning_rate : float
        beta_1, beta_2 : float
            Adam betas.
        clipnorm : float or None
            Per-parameter gradient norm clip.
        clipvalue : float or None
            Per-parameter gradient value clip.
        global_clipnorm : float or None
            Global gradient norm clip (passed to Lightning Trainer via clip_grad_norm).
        """
        super().__init__()
        self.prior = prior
        self.surrogate_posterior = surrogate_posterior
        self.likelihood = likelihood
        self.scaling_model = scaling_model
        self.mc_sample_size = mc_sample_size
        self.kl_weight = kl_weight
        self.scale_kl_weight = scale_kl_weight
        self.scale_prior = scale_prior

        self._learning_rate = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._clipnorm = clipnorm
        self._clipvalue = clipvalue
        self._global_clipnorm = global_clipnorm
        self._adam_epsilon = adam_epsilon
        self._filter_nan_gradients = filter_nan_gradients

        # Running history collected during train_model
        self._history = {}

    # ------------------------------------------------------------------
    # Keras-like add_kl_div helper
    # ------------------------------------------------------------------

    def add_kl_div(self, posterior, prior, samples=None, weight=1., reduction='sum',
                   name="KLDiv", metric_scale=1.):
        """
        Compute KL divergence (or MC estimate thereof), accumulate as a loss term,
        and register it as a named metric.

        Parameters
        ----------
        posterior, prior : distributions
            Distributions supporting log_prob; analytical KL attempted first,
            MC estimate used as fallback.
        samples : Tensor or None
            Pre-drawn samples from posterior for MC estimation.
        weight : float
            Multiplicative weight on the KL loss term.
        reduction : 'sum' | 'mean' | callable
            How to reduce the per-element KL before accumulation.
        name : str
            Metric name displayed during training.
        metric_scale : float
            Multiplier applied to the *reported metric* only (not the loss). Used
            during gradient accumulation so that a `mean`-reduced term evaluated on
            a mini-batch reports its share of the full-dataset value, which makes
            the per-batch metrics additive.
        """
        try:
            # Try analytical KL
            p_dist = posterior._distribution() if hasattr(posterior, '_distribution') else posterior
            q_dist = prior._distribution() if hasattr(prior, '_distribution') else prior
            kl_div = torch.distributions.kl_divergence(p_dist, q_dist)
        except NotImplementedError:
            # Fall back to MC estimate
            if samples is None:
                samples = posterior.rsample((self.mc_sample_size,))
            kl_div = posterior.log_prob(samples) - prior.log_prob(samples)

        if reduction == 'sum':
            kl_div = kl_div.sum() / self.mc_sample_size
        elif reduction == 'mean':
            kl_div = kl_div.mean()
        elif callable(reduction):
            kl_div = reduction(kl_div)

        self.add_loss(weight * kl_div)
        self.add_metric(metric_scale * kl_div, name)
        return kl_div

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def sample_structure_factors(self):
        """
        Draw one set of reparameterized samples from q(F).

        Returns
        -------
        z_f : Tensor, shape (mc_sample_size, n_structure_factors)
        """
        return self.surrogate_posterior.rsample((self.mc_sample_size,))

    def add_structure_factor_kl(self, z_f):
        """
        Accumulate the structure factor KL term, KL[q(F) || p(F)].

        This term ranges over the whole ASU collection rather than over
        observations, so it is independent of any batching of the reflection
        data and must be evaluated exactly once per gradient step.
        """
        if self.kl_weight is None:
            return self.add_kl_div(
                self.surrogate_posterior, self.prior, z_f,
                name='F KLDiv', reduction='sum'
            )
        return self.add_kl_div(
            self.surrogate_posterior, self.prior, z_f,
            weight=self.kl_weight, name='F KLDiv', reduction='mean'
        )

    def _rsample_scale(self, scale_dist, noise=None):
        """
        Reparameterized sample from the scale posterior q(Σ).

        Parameters
        ----------
        scale_dist : distribution
            Output of the scaling model.
        noise : Tensor or None
            Standard normal noise of shape (mc_sample_size, n_obs). When supplied,
            the sample is formed explicitly as ``loc + noise * scale`` rather than
            by calling ``rsample``. This is what ``Normal.rsample`` does internally;
            supplying the noise externally lets a gradient step reuse the identical
            random numbers no matter how the observations are split into batches.
        """
        if noise is None:
            return scale_dist.rsample((self.mc_sample_size,))

        loc = getattr(scale_dist, 'loc', None)
        scale = getattr(scale_dist, 'scale', None)
        if loc is None or scale is None:
            raise TypeError(
                f"{type(scale_dist).__name__} does not expose loc/scale, so externally "
                "supplied noise cannot be used. Pass deterministic_scale_noise=False."
            )
        return loc + noise * scale

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, inputs, z_f=None, scale_noise=None, batch_weight=1., add_f_kl=True):
        """
        Run one forward pass, accumulating loss/metric terms into the thread-local
        context. Call reset_losses_and_metrics() before invoking.

        Parameters
        ----------
        inputs : tuple of Tensors
            Reflection data, or a contiguous slice thereof.
        z_f : Tensor or None
            Pre-drawn samples from q(F), shape (mc_sample_size, n_structure_factors).
            When None (the default) samples are drawn here and the structure factor
            KL term is added, reproducing the unbatched behaviour. When supplied,
            the caller owns both the sampling and the KL term; this is how
            `train_model` shares a single set of structure factor samples across
            every batch of a gradient accumulation step.
        scale_noise : Tensor or None
            Standard normal noise for q(Σ) of shape (mc_sample_size, n_obs).
            See `_rsample_scale`.
        batch_weight : float
            Fraction of the full dataset contained in `inputs`. Applied to the
            `mean`-reduced loss terms so that summing over a partition of the data
            reproduces the whole-dataset mean. `sum`-reduced terms are additive
            already and are left alone.
        add_f_kl : bool
            Whether to add the structure factor KL term when `z_f` is drawn here.

        Returns
        -------
        ipred : Tensor, shape (mc_sample_size, n_obs)
            Predicted intensities for each MC sample and observation.
        """
        # Reparameterized samples from q(F) and q(Σ)
        owns_f_kl = z_f is None and add_f_kl
        if z_f is None:
            z_f = self.sample_structure_factors()
        # z_f: (mc_sample_size, n_refls)

        scale_dist = self.scaling_model(inputs)
        z_scale = self._rsample_scale(scale_dist, scale_noise)
        # z_scale: (mc_sample_size, n_obs)

        # Optional scale KL
        if self.scale_prior is not None:
            if self.scale_kl_weight is None:
                self.add_kl_div(
                    scale_dist, self.scale_prior, z_scale,
                    weight=self.scale_kl_weight,  # faithful reproduction of original (passes None)
                    reduction='sum', name="Σ KLDiv"
                )
            else:
                self.add_kl_div(
                    scale_dist, self.scale_prior, z_scale,
                    weight=batch_weight, reduction='mean', name="Σ KLDiv",
                    metric_scale=batch_weight,
                )

        refl_id = self.get_refl_id(inputs).squeeze(-1).long()

        # Predicted intensity: I = Σ * F²
        # NOTE: use embedding (not fancy indexing) for the gather — its backward uses a
        # sort+segment-reduce instead of atomic scatter-add, which matters a lot here
        # because refl_id has heavy duplication (many obs -> one reflection).
        f_gathered = torch.nn.functional.embedding(refl_id, z_f.t()).t()
        ipred = z_scale * f_gathered ** 2
        # ipred: (mc_sample_size, n_obs)

        likelihood = self.likelihood(inputs)
        ll = likelihood.log_prob(ipred)
        # ll: (mc_sample_size, n_obs)

        # Structure factor KL and log likelihood reduction
        if self.kl_weight is None:
            if owns_f_kl:
                self.add_structure_factor_kl(z_f)
            ll = ll.sum() / self.mc_sample_size
        else:
            if owns_f_kl:
                self.add_structure_factor_kl(z_f)
            ll = ll.mean() * batch_weight

        self.add_loss(-ll)
        self.add_metric(-ll, "NLL")

        return ipred

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        reset_losses_and_metrics()
        self(batch)
        losses = get_accumulated_losses()
        metrics = get_accumulated_metrics()

        loss = sum(losses)
        for name, val in metrics.items():
            self.log(name, val, prog_bar=True, on_step=True, on_epoch=False)

        # Gradient norm (computed by Lightning trainer after this step)
        return loss

    def configure_optimizers(self):
        opt = AdamEpsInsideSqrt(
            self.parameters(),
            lr=self._learning_rate,
            betas=(self._beta_1, self._beta_2),
            eps=self._adam_epsilon,
        )
        return opt

    # ------------------------------------------------------------------
    # Data caching and batching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_inputs(data, device):
        """
        Materialize the full dataset on `device` once, up front.

        Batches are then contiguous slices of these resident tensors, so the inner
        training loop never touches the host. float64 arrays are demoted to float32
        (matching the dtype the models compute in); integer index arrays keep their
        dtype.

        Parameters
        ----------
        data : sequence of array-like
            Careless input tuple (numpy arrays or tensors).
        device : torch.device

        Returns
        -------
        tuple of Tensors, contiguous and resident on `device`.
        """
        cached = []
        for d in data:
            t = d if isinstance(d, torch.Tensor) else torch.as_tensor(d)
            if t.dtype == torch.float64:
                t = t.to(torch.float32)
            cached.append(t.to(device).contiguous())
        return tuple(cached)

    @staticmethod
    def _slice_inputs(inputs, lo, hi):
        """Take rows [lo, hi) of every entry of an input tuple. Slices are views."""
        return tuple(t[lo:hi] for t in inputs)

    @staticmethod
    def _batch_boundaries(inputs, num_batches):
        """
        Partition the observations into `num_batches` contiguous [lo, hi) ranges.

        Contiguous ranges are used because slicing is a free view of the cached
        tensors, and because a partition guarantees every observation contributes
        exactly once, so the accumulated gradient equals the whole-dataset gradient.

        For Laue data the likelihood convolves predictions within a `harmonic_id`
        group, so a group must never straddle a batch boundary. Groups are merged
        into indivisible segments (the classic "partition labels" scan over each
        label's last occurrence) and boundaries are snapped to segment ends. If the
        harmonics interleave so heavily that fewer than `num_batches` segments
        exist, fewer batches are returned.

        Returns
        -------
        list of (lo, hi) tuples covering [0, n_obs).
        """
        n = int(BaseModel.get_refl_id(inputs).shape[0])
        if num_batches <= 1 or n == 0:
            return [(0, n)]

        ends = None
        if BaseModel.is_laue(inputs):
            harmonic_id = BaseModel.get_harmonic_id(inputs).squeeze(-1).long()
            position = torch.arange(n, device=harmonic_id.device)
            last = torch.zeros(
                int(harmonic_id.max()) + 1, dtype=torch.long, device=harmonic_id.device
            )
            # Last occurrence of each harmonic group. scatter_reduce with 'amax' is
            # used rather than `last[harmonic_id] = position`, because indexed
            # assignment with duplicate indices is nondeterministic on CUDA: an
            # arbitrary write wins, which would understate a group's extent and let
            # a boundary split it.
            last.scatter_reduce_(0, harmonic_id, position, reduce='amax')
            reach = torch.cummax(last[harmonic_id], dim=0).values
            ends = (torch.nonzero(reach == position).squeeze(-1) + 1).cpu()

        boundaries = []
        lo = 0
        for k in range(1, num_batches):
            target = (k * n) // num_batches
            if ends is None:
                hi = target
            else:
                idx = int(torch.searchsorted(ends, torch.tensor(target)))
                hi = int(ends[idx]) if idx < len(ends) else n
            if hi > lo and hi < n:
                boundaries.append((lo, hi))
                lo = hi
        boundaries.append((lo, n))
        return boundaries

    @staticmethod
    def _read_metrics(metrics):
        """
        Read every metric back to the host in a single transfer.

        Metrics accumulate as 0-dim device tensors during the step. Reading them one
        at a time -- what float() or .item() does -- stalls the launch queue once per
        metric, and under gradient accumulation once per metric *per batch*. Stacking
        them makes it one stall no matter how many there are.
        """
        tensors = {k: v for k, v in metrics.items() if torch.is_tensor(v)}
        values = {k: float(v) for k, v in metrics.items() if not torch.is_tensor(v)}
        if tensors:
            read = torch.stack(
                [v.detach().reshape(()).float() for v in tensors.values()]
            ).tolist()
            values.update(zip(tensors.keys(), read))
        # Preserve insertion order, which is what the progress bar and history show.
        return {k: values[k] for k in metrics}

    @staticmethod
    def _accumulate_metrics(target, new):
        """
        Sum per-batch metrics into `target`.

        Every batch-dependent metric is already scaled to that batch's share of the
        full dataset (see `forward`'s `batch_weight`), so summation reconstructs the
        whole-dataset value. Metrics emitted once per step (F KLDiv, rDW_i) appear
        in a single call and pass through unchanged.

        Values stay on the accelerator as 0-dim tensors. Calling float() here would
        cost one host synchronization per metric per batch -- with six metrics and
        eight batches that is ~48 stalls a step -- so the whole set is read once, in
        one transfer, at the end of the step.
        """
        for k, v in new.items():
            target[k] = target.get(k, 0.) + v
        return target

    # ------------------------------------------------------------------
    # Custom training loop (mirrors original train_model API)
    # ------------------------------------------------------------------

    @staticmethod
    def _torch_compile_kwargs(jit_compile_mode, reduce_retracing):
        """
        Build the torch.compile keyword arguments for a given mode.

        The name matters. lightning patches torch.compile globally
        (lightning.fabric.wrappers._capture_compile_kwargs) so that it writes a
        ``_compile_kwargs`` dict onto the module it returns, and
        OptimizedModule.__setattr__ forwards any unknown attribute to _orig_mod --
        so after one compile, ``model._compile_kwargs`` is a dict on *this* model.
        A helper called ``_compile_kwargs`` would be shadowed by it, and the second
        train_model call on the same instance would die with "'dict' object is not
        callable".

        Raises on the one combination that is known to be broken: handing a
        dynamic-shape graph to CUDA graphs segfaults the process (SIGSEGV
        immediately after compilation, torch 2.13 / triton 3.7, reproduced on
        both CUDA-graphs modes). Failing loudly here beats a bare crash with no
        traceback partway into a merge.
        """
        from careless.args.tf_options import CUDA_GRAPH_MODES, JIT_COMPILE_MODES

        if jit_compile_mode not in JIT_COMPILE_MODES:
            raise ValueError(
                f"Unknown jit_compile_mode {jit_compile_mode!r}; "
                f"expected one of {list(JIT_COMPILE_MODES)}"
            )
        if reduce_retracing and jit_compile_mode in CUDA_GRAPH_MODES:
            raise ValueError(
                f"--reduce-retracing cannot be combined with "
                f"--jit-compile-mode={jit_compile_mode}, which uses CUDA graphs: the "
                f"combination segfaults. Use --jit-compile-mode=max-autotune-no-cudagraphs "
                f"(the default, and the faster mode anyway) or drop --reduce-retracing."
            )

        kwargs = {"dynamic": reduce_retracing}
        if jit_compile_mode != "default":
            kwargs["mode"] = jit_compile_mode
        return kwargs


    def train_model(
        self,
        data,
        steps,
        message=None,
        format_string="{:0.2e}",
        validation_data=None,
        validation_frequency=10,
        progress=True,
        num_batches=1,
        deterministic_scale_noise=True,
        jit_compile=None,
        jit_compile_mode="max-autotune-no-cudagraphs",
        reduce_retracing=False,
    ):
        """
        Train using a simple manual loop with optional gradient accumulation.
        Returns a history dict with one entry per step.

        With `num_batches > 1` the reflection data is split into that many
        contiguous mini-batches. Each is forward/backward-ed in turn and the
        gradients accumulate into `.grad` before a single optimizer step, so peak
        activation memory falls roughly as 1/num_batches while the update itself is
        unchanged.

        Two details make the accumulated step equal to the whole-dataset step:

        * The structure factors are sampled **once** per step, before the batch
          loop, and the same `z_f` is used for every batch's likelihood. To keep
          that single sample's graph alive across several `backward()` calls, the
          chain rule is split at the sample: batches differentiate a detached copy
          and accumulate dL/dz_f, which is then pushed back through the sampler
          together with the structure factor KL term in one final backward.
        * The whole dataset is cached on the accelerator before the loop starts, so
          a batch is a zero-copy view and no host/device transfer happens per step.

        Parameters
        ----------
        data : tuple of Tensors
            Full dataset as a tuple of arrays or tensors.
        steps : int
            Number of gradient steps.
        message : str, optional
            Description shown in progress bar.
        format_string : str
            Format string for metric display.
        validation_data : tuple or None
            Optional validation tensors evaluated every validation_frequency steps.
        validation_frequency : int
            Evaluate validation_data every this many steps.
        progress : bool
            Whether to display a tqdm progress bar.
        num_batches : int
            Number of gradient accumulation batches per step. 1 (default) reproduces
            the original whole-dataset step.
        deterministic_scale_noise : bool
            Pre-draw the standard normal noise used to sample q(Σ) for the whole
            dataset once per step and slice it per batch, instead of sampling inside
            each batch. This costs one float32 per observation per MC sample and
            makes a step bit-for-bit reproducible across any value of `num_batches`.
        jit_compile : bool, optional
            If truthy, wrap the forward pass with torch.compile.
        jit_compile_mode : str
            The torch.compile mode to use when jit_compile is truthy. One of
            careless.args.tf_options.JIT_COMPILE_MODES. The default,
            "max-autotune-no-cudagraphs", was the fastest and least memory hungry
            of the four on the window-merge benchmark; see doc/performance/.
        reduce_retracing : bool
            If True, allow dynamic shapes in torch.compile to avoid recompilation.
            Cannot be combined with a CUDA-graphs jit_compile_mode.
        """
        from tqdm import trange

        # Allow TF32 on Ampere+ GPUs for faster matmuls in the scaling MLP
        torch.set_float32_matmul_precision('high')

        optimizer = self.configure_optimizers()
        history = {}

        forward_fn = self
        if jit_compile:
            forward_fn = torch.compile(
                self, **self._torch_compile_kwargs(jit_compile_mode, reduce_retracing)
            )

        # Cache the full dataset on the accelerator once, before optimization.
        device = next(self.parameters()).device
        data = self._cache_inputs(data, device)
        n_obs = int(self.get_refl_id(data).shape[0])

        if validation_data is not None:
            n_val = int(len(validation_data[0]))
            val_scale = n_obs / n_val
            validation_data = self._cache_inputs(validation_data, device)

        # A one-row batch would be squashed by BaseModel.get_input_by_name's
        # leading-singleton squeeze, so keep every batch at least two rows.
        num_batches = max(1, min(int(num_batches), max(1, n_obs // 2)))
        batches = self._batch_boundaries(data, num_batches)
        val_batches = None
        if validation_data is not None:
            val_batches = self._batch_boundaries(
                validation_data, max(1, min(num_batches, max(1, n_val // 2)))
            )

        if len(batches) > 1:
            from careless.models.likelihoods.mono import NeuralLikelihood
            if isinstance(self.likelihood, NeuralLikelihood):
                from warnings import warn
                warn(
                    "NeuralLikelihood normalizes uncertainties by a batch mean, so its "
                    "gradients depend on how the data are batched. Gradient accumulation "
                    "is not equivalent to a whole-dataset step for this likelihood.",
                    RuntimeWarning,
                )

        # Reusable noise buffers for q(Σ); filled in place each step.
        noise = val_noise = None
        if deterministic_scale_noise:
            noise = torch.empty(
                (self.mc_sample_size, n_obs), dtype=torch.float32, device=device
            )
            if validation_data is not None:
                val_noise = torch.empty(
                    (self.mc_sample_size, n_val), dtype=torch.float32, device=device
                )

        bar = trange(steps, desc=message, disable=not progress)
        for i in bar:
            self.train()
            optimizer.zero_grad()
            if noise is not None:
                noise.normal_()

            # Sample the structure factors exactly once for the whole step, then
            # cut the graph so each batch can be backward-ed independently.
            z_f = self.sample_structure_factors()
            z_f_batch = z_f.detach().requires_grad_(True)

            # Build the structure factor KL term now (it spans the whole ASU and is
            # independent of the data batching); its backward waits for dL/dz_f.
            reset_losses_and_metrics()
            self.add_structure_factor_kl(z_f)
            kl_loss = sum(get_accumulated_losses())
            metrics = self._accumulate_metrics({}, get_accumulated_metrics())

            # The running loss stays a 0-dim device tensor. The original
            # accumulation branch called float() on it once per batch and tested
            # torch.isfinite per batch, which is three host stalls per batch and
            # therefore a cost that grows with num_batches. Divergence is instead
            # checked once per step, below, from the same transfer that reads the
            # metrics -- and still before optimizer.step(), so a bad gradient is
            # never applied. A batch that produces NaN now runs its backward before
            # the check sees it, which dirties .grad; that is harmless because the
            # step is abandoned and zero_grad() precedes every step.
            loss = kl_loss.detach()

            for lo, hi in batches:
                reset_losses_and_metrics()
                forward_fn(
                    self._slice_inputs(data, lo, hi),
                    z_f=z_f_batch,
                    scale_noise=None if noise is None else noise[:, lo:hi],
                    batch_weight=(hi - lo) / n_obs,
                )
                batch_loss = sum(get_accumulated_losses())
                self._accumulate_metrics(metrics, get_accumulated_metrics())

                if batch_loss.requires_grad:
                    batch_loss.backward()
                loss = loss + batch_loss.detach()

            # One backward for everything that flows through the shared samples:
            # the KL term directly, and the accumulated dL/dz_f from every batch.
            # Either root can be constant when its parameters are frozen
            # (--freeze-structure-factors), in which case it is simply dropped.
            roots, grads = [], []
            if kl_loss.requires_grad:
                roots.append(kl_loss)
                grads.append(torch.ones_like(kl_loss))
            if z_f.requires_grad and z_f_batch.grad is not None:
                roots.append(z_f)
                grads.append(z_f_batch.grad)
            if roots:
                torch.autograd.backward(tuple(roots), tuple(grads))

            metrics["Loss"] = loss

            # Per-element NaN/Inf gradient filter (matches TF behaviour)
            if self._filter_nan_gradients:
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            # Gradient clipping
            if self._global_clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self._global_clipnorm)
            if self._clipnorm is not None:
                for p in self.parameters():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_([p], self._clipnorm)
            if self._clipvalue is not None:
                torch.nn.utils.clip_grad_value_(self.parameters(), self._clipvalue)

            # Compute grad norm for monitoring
            grad_norm = torch.sqrt(
                sum(p.grad.norm() ** 2 for p in self.parameters() if p.grad is not None)
            )

            metrics["Grad Norm"] = grad_norm

            # One transfer for the whole step: every metric, the loss and the grad
            # norm come back together. It doubles as the divergence check, and it
            # happens before optimizer.step(), so a non-finite step is still never
            # applied. Its cost does not grow with num_batches, which is the point:
            # the pre-merge accumulation loop paid three stalls per batch.
            values = self._read_metrics(metrics)

            if not np.isfinite(values["Loss"]):
                print("Encountered numerical issues, terminating optimization early!")
                break

            optimizer.step()

            # Validation
            if validation_data is not None:
                if i % validation_frequency == 0:
                    self.eval()
                    with torch.no_grad():
                        if val_noise is not None:
                            val_noise.normal_()
                        z_f_val = self.sample_structure_factors()
                        val_metrics = {}
                        for lo, hi in val_batches:
                            reset_losses_and_metrics()
                            forward_fn(
                                self._slice_inputs(validation_data, lo, hi),
                                z_f=z_f_val,
                                scale_noise=None if val_noise is None else val_noise[:, lo:hi],
                                batch_weight=(hi - lo) / n_val,
                            )
                            self._accumulate_metrics(val_metrics, get_accumulated_metrics())
                    nll_val = val_metrics.get("NLL")
                    # A second transfer, but only on validation steps.
                    values["NLL_val"] = (
                        float('nan') if nll_val is None else float(nll_val) * val_scale
                    )
                else:
                    values["NLL_val"] = float('nan')

            postfix = {}
            for k, v in values.items():
                postfix[k] = format_string.format(v)
                history.setdefault(k, []).append(v)
            bar.set_postfix(postfix)

        return history

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def scale_moments(self, inputs, num_batches=1):
        """
        Per-observation mean and standard deviation of q(Sigma), evaluated in
        `num_batches` contiguous chunks and returned at full length.

        Only the scaling model is chunked, and deliberately so. `ImageLayer` gathers
        an (n_obs, width, width) weight tensor, which makes this the one place where
        inference memory grows with width**2 -- n_obs * width**2 * 4 bytes is 35 GiB
        at width 48 on a 4.1M-reflection dataset. Everything downstream of it is
        O(n_obs). Assembling full-length results here means the Laue convolution,
        which ranges over harmonic groups and indexes by a global harmonic_id, still
        sees the whole array exactly as it did before.

        Returns
        -------
        mean, stddev : Tensors of shape (n_obs,)
        """
        n_obs = int(self.get_refl_id(inputs).shape[0])
        # A one-row chunk would be squashed by BaseModel.get_input_by_name's
        # leading-singleton squeeze, so keep every chunk at least two rows.
        num_batches = max(1, min(int(num_batches), max(1, n_obs // 2)))
        if num_batches == 1:
            dist = self.scaling_model(inputs)
            return dist.mean.detach(), dist.stddev.detach()

        mean = stddev = None
        for lo, hi in self._batch_boundaries(inputs, num_batches):
            dist = self.scaling_model(self._slice_inputs(inputs, lo, hi))
            chunk_mean, chunk_std = dist.mean.detach(), dist.stddev.detach()
            if mean is None:
                mean = torch.empty(n_obs, dtype=chunk_mean.dtype, device=chunk_mean.device)
                stddev = torch.empty(n_obs, dtype=chunk_std.dtype, device=chunk_std.device)
            mean[lo:hi] = chunk_mean
            stddev[lo:hi] = chunk_std
        return mean, stddev

    @torch.no_grad()
    def scale_mean_stddev(self, inputs, num_batches=1):
        """
        Compute mean and standard deviation of the scale posterior for each observation.

        Parameters
        ----------
        num_batches : int
            Evaluate the scaling model in this many contiguous chunks. See
            `scale_moments`; 1 (default) reproduces the whole-dataset call.

        Returns
        -------
        mean : np.ndarray
        stddev : np.ndarray
        """
        device = next(self.parameters()).device
        mean_t, stddev_t = self.scale_moments(inputs, num_batches)
        mean = mean_t.cpu().numpy()
        stddev = stddev_t.cpu().numpy()

        from careless.models.likelihoods.laue import LaueBase
        if isinstance(self.likelihood, LaueBase):
            likelihood = self.likelihood(inputs)
            mean = likelihood.convolve(torch.as_tensor(mean, device=device)).cpu().numpy()
            stddev = np.sqrt(
                likelihood.convolve(torch.as_tensor(stddev ** 2, device=device)).cpu().numpy()
            )

        return mean, stddev

    @torch.no_grad()
    def prediction_mean_stddev(self, inputs, num_batches=1):
        """
        Compute mean and standard deviation of the predicted intensity E[I].

        Parameters
        ----------
        num_batches : int
            Evaluate the scaling model in this many contiguous chunks. See
            `scale_moments`; 1 (default) reproduces the whole-dataset call.

        Returns
        -------
        mean : np.ndarray
        stddev : np.ndarray
        """
        device = next(self.parameters()).device
        refl_id = self.get_refl_id(inputs).squeeze(-1).long()
        scale_mean, scale_stddev = self.scale_moments(inputs, num_batches)

        F_mean = self.surrogate_posterior.mean.detach()
        F_std = self.surrogate_posterior.stddev.detach()

        # <I> = <Sigma> * (<F^2>) = <Sigma> * (Var(F) + <F>^2)
        f2 = F_std ** 2 + F_mean ** 2
        iexp = scale_mean * f2[refl_id]
        iexp = iexp.cpu().numpy()

        # var(I) = <I^2> - <I>^2 = <F^4><Sigma^2> - <I>^2
        f4 = torch.as_tensor(
            self.surrogate_posterior.moment_4(method='scipy'), dtype=torch.float32, device=device
        )
        s2 = scale_mean ** 2 + scale_stddev ** 2
        ivar = f4[refl_id] * s2 - torch.as_tensor(iexp, device=device) ** 2
        ivar = ivar.cpu().numpy()

        from careless.models.likelihoods.laue import LaueBase
        if isinstance(self.likelihood, LaueBase):
            likelihood = self.likelihood(inputs)
            iexp_t = torch.as_tensor(iexp, device=device)
            ivar_t = torch.as_tensor(ivar, device=device)
            iexp = likelihood.convolve(iexp_t).cpu().numpy()
            ivar = likelihood.convolve(ivar_t).cpu().numpy()

        return iexp, np.sqrt(np.maximum(ivar, 0.0))
