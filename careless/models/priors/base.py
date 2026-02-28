from careless.models.base import BaseModel


class Prior(BaseModel):
    """Base class for prior distributions on merged normalized structure factor amplitudes."""

    def log_prob(self, x):
        raise NotImplementedError(
            "All Prior subclasses must implement log_prob(x)."
        )
