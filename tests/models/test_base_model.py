import pytest
from careless.models.base import BaseModel



def test_is_laue(laue_inputs, mono_inputs):
    assert BaseModel.is_laue(laue_inputs)
    assert not BaseModel.is_laue(mono_inputs)

def test_getters(laue_inputs, mono_inputs):
    for inputs in laue_inputs,mono_inputs:
        if BaseModel.is_laue(inputs):
            BaseModel.get_harmonic_id(inputs)
            BaseModel.get_wavelength(inputs)
        BaseModel.get_image_id(inputs)
        BaseModel.get_intensities(inputs)
        BaseModel.get_metadata(inputs)
        BaseModel.get_refl_id(inputs)
        BaseModel.get_uncertainties(inputs)

def test_by_name():
    for k,v in BaseModel.input_index.items():
        assert BaseModel.get_name_by_index(v) == k

def test_by_index():
    for k,v in BaseModel.input_index.items():
        assert BaseModel.get_index_by_name(k) == v

