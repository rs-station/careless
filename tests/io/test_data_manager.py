import pytest
import numpy as np
from careless.io.manager import DataManager
from careless.models.base import BaseModel


def test_data_manager_laue(laue_inputs, laue_reciprocal_asu_collection):
    inputs = laue_inputs
    rac = laue_reciprocal_asu_collection

    dm = DataManager(inputs, rac)
    dm.get_tf_dataset()
    dm.get_tf_dataset(inputs)
    train,test = dm.split_data_by_refl(0.1)
    train,test = dm.split_data_by_image(0.1)
    q = dm.get_wilson_prior()
    _ = dm.get_wilson_prior(20.)
    results = dm.get_results(q)
    for result in results:
        assert result.N.min() > 0

def test_data_manager_mono(mono_inputs, mono_reciprocal_asu_collection):
    inputs = mono_inputs
    rac = mono_reciprocal_asu_collection

    dm = DataManager(inputs, rac)
    dm.get_tf_dataset()
    dm.get_tf_dataset(inputs)
    train,test = dm.split_data_by_refl(0.1)
    train,test = dm.split_data_by_image(0.1)
    q = dm.get_wilson_prior()
    _ = dm.get_wilson_prior(20.)
    results = dm.get_results(q)
    for result in results:
        assert result.N.min() > 0
