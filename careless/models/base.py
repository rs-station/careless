import threading
import torch
import torch.nn as nn
import numpy as np


# Thread-local context for accumulating losses and metrics during forward passes,
# mirroring the Keras add_loss / add_metric idiom.
_loss_ctx = threading.local()


def _get_losses():
    return getattr(_loss_ctx, 'losses', None)


def _get_metrics():
    return getattr(_loss_ctx, 'metrics', None)


def reset_losses_and_metrics():
    _loss_ctx.losses = []
    _loss_ctx.metrics = {}


def get_accumulated_losses():
    return list(getattr(_loss_ctx, 'losses', []))


def get_accumulated_metrics():
    return dict(getattr(_loss_ctx, 'metrics', {}))


class BaseModel(nn.Module):
    """
    Base class for all models in `careless`.
    Encodes accessors for the standard format inputs and provides Keras-style
    add_loss / add_metric accumulation via thread-local storage.

    Input tuple ordering:
        [refl_id, image_id, file_id, metadata, intensities, uncertainties]
        [refl_id, image_id, file_id, metadata, intensities, uncertainties, wavelength, harmonic_id]  (Laue)
    """
    input_index = {
        'refl_id'       : 0,
        'image_id'      : 1,
        'file_id'       : 2,
        'metadata'      : 3,
        'intensities'   : 4,
        'uncertainties' : 5,
        'wavelength'    : 6,
        'harmonic_id'   : 7,
    }

    def add_loss(self, loss):
        """Accumulate a scalar loss term (mirrors keras Model.add_loss)."""
        losses = _get_losses()
        if losses is not None:
            losses.append(loss)

    def add_metric(self, value, name):
        """Accumulate a named metric (mirrors keras Model.add_metric)."""
        metrics = _get_metrics()
        if metrics is not None:
            metrics[name] = value.detach() if hasattr(value, 'detach') else value

    @staticmethod
    def is_laue(inputs: tuple) -> bool:
        laue_size = BaseModel.input_index['harmonic_id'] + 1
        return len(inputs) >= laue_size

    @staticmethod
    def get_name_by_index(index: int) -> str:
        for k, v in BaseModel.input_index.items():
            if v == index:
                return k
        raise ValueError(
            f"index {index} not valid. Valid indices: {list(BaseModel.input_index.values())}"
        )

    @staticmethod
    def get_index_by_name(name):
        if name not in BaseModel.input_index:
            raise ValueError(
                f"name '{name}' not valid. Valid keys: {list(BaseModel.input_index.keys())}"
            )
        return BaseModel.input_index[name]

    @staticmethod
    def get_input_by_name(inputs, name):
        if name not in BaseModel.input_index:
            raise ValueError(
                f"name '{name}' not valid. Valid keys: {list(BaseModel.input_index.keys())}"
            )
        idx = BaseModel.input_index[name]
        try:
            datum = inputs[idx]
        except Exception:
            raise ValueError(
                f"Attempting to gather '{name}' from inputs with length {len(inputs)} failed."
            )
        if isinstance(datum, torch.Tensor) and datum.shape[0] == 1:
            datum = datum.squeeze(0)
        elif isinstance(datum, np.ndarray) and datum.shape[0] == 1:
            datum = datum.squeeze(0)
        return datum

    @staticmethod
    def get_refl_id(inputs):
        return BaseModel.get_input_by_name(inputs, 'refl_id')

    @staticmethod
    def get_file_id(inputs):
        return BaseModel.get_input_by_name(inputs, 'file_id')

    @staticmethod
    def get_image_id(inputs):
        return BaseModel.get_input_by_name(inputs, 'image_id')

    @staticmethod
    def get_metadata(inputs):
        return BaseModel.get_input_by_name(inputs, 'metadata')

    @staticmethod
    def get_intensities(inputs):
        return BaseModel.get_input_by_name(inputs, 'intensities')

    @staticmethod
    def get_uncertainties(inputs):
        return BaseModel.get_input_by_name(inputs, 'uncertainties')

    @staticmethod
    def get_wavelength(inputs):
        return BaseModel.get_input_by_name(inputs, 'wavelength')

    @staticmethod
    def get_harmonic_id(inputs):
        return BaseModel.get_input_by_name(inputs, 'harmonic_id')
