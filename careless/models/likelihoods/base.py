from careless.models.base import BaseModel


class Likelihood(BaseModel):
    """
    Base class for observation likelihoods.
    Subclasses must implement forward(inputs) returning an object with a log_prob method.
    """

    def forward(self, inputs):
        raise NotImplementedError(
            "Likelihoods must implement forward(inputs) returning a distribution "
            "with a log_prob(x) method."
        )
