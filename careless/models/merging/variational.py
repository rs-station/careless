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

    def add_kl_div(self, posterior, prior, samples=None, weight=1., reduction='sum', name="KLDiv"):
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
        self.add_metric(kl_div, name)
        return kl_div

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, inputs):
        """
        Run one forward pass, accumulating loss/metric terms into the thread-local
        context. Call reset_losses_and_metrics() before invoking.

        Returns
        -------
        ipred : Tensor, shape (mc_sample_size, n_obs)
            Predicted intensities for each MC sample and observation.
        """
        # Reparameterized samples from q(F) and q(Σ)
        z_f = self.surrogate_posterior.rsample((self.mc_sample_size,))
        # z_f: (mc_sample_size, n_refls)

        scale_dist = self.scaling_model(inputs)
        z_scale = scale_dist.rsample((self.mc_sample_size,))
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
                    weight=1.0, reduction='mean', name="Σ KLDiv"
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
            self.add_kl_div(
                self.surrogate_posterior, self.prior, z_f,
                name='F KLDiv', reduction='sum'
            )
            ll = ll.sum() / self.mc_sample_size
        else:
            self.add_kl_div(
                self.surrogate_posterior, self.prior, z_f,
                weight=self.kl_weight, name='F KLDiv', reduction='mean'
            )
            ll = ll.mean()

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
            self.log(name, float(val), prog_bar=True, on_step=True, on_epoch=False)

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
    # Custom training loop (mirrors original train_model API)
    # ------------------------------------------------------------------

    def train_model(
        self,
        data,
        steps,
        message=None,
        format_string="{:0.2e}",
        validation_data=None,
        validation_frequency=10,
        progress=True,
        jit_compile=None,
        reduce_retracing=False,
    ):
        """
        Train using a simple manual loop (whole-dataset batching).
        Returns a history dict with one entry per step.

        Parameters
        ----------
        data : tuple of Tensors
            Full dataset as a tuple of tensors (already on device).
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
        jit_compile : bool, optional
            If truthy, wrap the forward pass with torch.compile.
        reduce_retracing : bool
            If True, allow dynamic shapes in torch.compile to avoid recompilation.
        """
        from tqdm import trange

        # Allow TF32 on Ampere+ GPUs for faster matmuls in the scaling MLP
        torch.set_float32_matmul_precision('high')

        optimizer = self.configure_optimizers()
        history = {}

        forward_fn = torch.compile(self, dynamic=reduce_retracing) if jit_compile else self

        # Move data to model's device
        device = next(self.parameters()).device
        data = tuple(
            torch.as_tensor(d, dtype=torch.float32).to(device)
            if d.dtype in (torch.float64, np.float64)
            else torch.as_tensor(d).to(device)
            for d in data
        )
        if validation_data is not None:
            val_scale = len(data[0]) / len(validation_data[0])
            validation_data = tuple(
                torch.as_tensor(d).to(device) for d in validation_data
            )

        bar = trange(steps, desc=message, disable=not progress)
        for i in bar:
            self.train()
            optimizer.zero_grad()
            reset_losses_and_metrics()

            forward_fn(data)

            losses = get_accumulated_losses()
            metrics = get_accumulated_metrics()
            loss = sum(losses)
            metrics["Loss"] = loss.detach().item()

            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print("Encountered numerical issues, terminating optimization early!")
                break

            loss.backward()

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

            optimizer.step()

            # Validation
            if validation_data is not None:
                if i % validation_frequency == 0:
                    self.eval()
                    with torch.no_grad():
                        reset_losses_and_metrics()
                        forward_fn(validation_data)
                        val_metrics = get_accumulated_metrics()
                    metrics["NLL_val"] = float(val_metrics.get("NLL", float('nan'))) * val_scale
                else:
                    metrics["NLL_val"] = float('nan')

            metrics["Grad Norm"] = float(grad_norm)

            # Update history
            postfix = {}
            for k, v in metrics.items():
                v = float(v)
                postfix[k] = format_string.format(v)
                history.setdefault(k, []).append(v)
            bar.set_postfix(postfix)

        return history

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def scale_mean_stddev(self, inputs):
        """
        Compute mean and standard deviation of the scale posterior for each observation.

        Returns
        -------
        mean : np.ndarray
        stddev : np.ndarray
        """
        device = next(self.parameters()).device
        scale_dist = self.scaling_model(inputs)
        mean = scale_dist.mean.detach().cpu().numpy()
        stddev = scale_dist.stddev.detach().cpu().numpy()

        from careless.models.likelihoods.laue import LaueBase
        if isinstance(self.likelihood, LaueBase):
            likelihood = self.likelihood(inputs)
            mean = likelihood.convolve(torch.as_tensor(mean, device=device)).cpu().numpy()
            stddev = np.sqrt(
                likelihood.convolve(torch.as_tensor(stddev ** 2, device=device)).cpu().numpy()
            )

        return mean, stddev

    @torch.no_grad()
    def prediction_mean_stddev(self, inputs):
        """
        Compute mean and standard deviation of the predicted intensity E[I].

        Returns
        -------
        mean : np.ndarray
        stddev : np.ndarray
        """
        device = next(self.parameters()).device
        refl_id = self.get_refl_id(inputs).squeeze(-1).long()
        scale_dist = self.scaling_model(inputs)

        F_mean = self.surrogate_posterior.mean.detach()
        F_std = self.surrogate_posterior.stddev.detach()

        # <I> = <Σ> * (<F²>) = <Σ> * (Var(F) + <F>²)
        f2 = F_std ** 2 + F_mean ** 2
        iexp = scale_dist.mean * f2[refl_id]
        iexp = iexp.cpu().numpy()

        # var(I) = <I²> - <I>² = <F⁴><Σ²> - <I>²
        f4 = torch.as_tensor(
            self.surrogate_posterior.moment_4(method='scipy'), dtype=torch.float32, device=device
        )
        s2 = scale_dist.mean ** 2 + scale_dist.stddev ** 2
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
