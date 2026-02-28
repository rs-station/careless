from careless.models.base import BaseModel


class Scaler(BaseModel):
    """Base class for scaling models."""

    def forward(self, inputs):
        raise NotImplementedError(
            "Scaler subclasses must implement forward(inputs) returning a distribution."
        )
