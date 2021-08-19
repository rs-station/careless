from careless.models.base import BaseModel


class Likelihood(BaseModel):
    def call(inputs):
        raise NotImplementedError(
            "Likelihoods must implement a call method that returns a `tfp.distribution.Distribution` "
            "or a similar object with a `log_prob` method."
        )

