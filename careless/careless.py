#!/usr/bin/env python

# This needs to be called before tf is imported 
# otherwise the log level will not be set properly

import tensorflow as tf
from careless.io.manager import DataManager
from careless.io.formatter import MonoFormatter,LaueFormatter
from careless.models.base import BaseModel
from careless.models.merging.surrogate_posteriors import TruncatedNormal
from careless.models.merging.variational import VariationalMergingModel
from careless.models.scaling.image import HybridImageScaler,ImageScaler
from careless.models.scaling.nn import MLPScaler
from careless.io.asu import *

def main():
    from careless.parser import parser
    parser = parser.parse_args()
    run_careless(parser)

def run_careless(parser):
    if parser.type == 'poly':
        df = LaueFormatter.from_parser(parser)
        from careless.models.likelihoods.laue import NormalLikelihood,StudentTLikelihood
    elif parser.type == 'mono':
        from careless.models.likelihoods.mono import NormalLikelihood,StudentTLikelihood
        df = MonoFormatter.from_parser(parser)


    inputs,rac = df.format_files(parser.reflection_files)
    n_images = np.max(BaseModel.get_image_id(inputs)) + 1
    dm = DataManager(inputs, rac)

    if parser.test_fraction is not None:
        train,test = dm.split_data_by_refl(parser.test_fraction)
    else:
        train,test = dm.inputs,None

    prior = dm.get_wilson_prior(parser.wilson_prior_b)
    loc,scale = prior.mean(),prior.stddev()/10.
    low = (1e-32 * rac.centric).astype('float32')
    surrogate_posterior = TruncatedNormal.from_loc_and_scale(loc, scale, low)
    dof = parser.studentt_likelihood_dof
    if dof is None:
        likelihood = NormalLikelihood()
    else:
        likelihood = StudentTLikelihood(dof)

    mlp_width = parser.mlp_width
    if mlp_width is None:
        mlp_width = BaseModel.get_metadata(inputs).shape[-1]
    mlp_scaler = MLPScaler(parser.mlp_layers, mlp_width)
    image_scaler = ImageScaler(n_images)
    scaler = HybridImageScaler(mlp_scaler, image_scaler)

    model = VariationalMergingModel(surrogate_posterior, prior, likelihood, scaler, parser.mc_samples)

    opt = tf.keras.optimizers.Adam(
        parser.learning_rate,
        parser.beta_1,
        parser.beta_2,
    )

    model.compile(opt)

    from careless.callbacks.progress_bar import ProgressBar
    callbacks = [
        ProgressBar(),
    ]

    model.fit(train, epochs=parser.iterations, steps_per_epoch=1, verbose=0, callbacks=callbacks,  shuffle=False)

    for i,ds in enumerate(dm.get_results(surrogate_posterior)):
        filename = parser.output_base + f'_{i}.mtz'
        ds.write_mtz(filename)


    if test is not None:
        for i,ds in enumerate(dm.get_predictions(model, test)):
            filename = parser.output_base + f'_test_{i}.mtz'
            ds.write_mtz(filename)
        for i,ds in enumerate(dm.get_predictions(model, train)):
            filename = parser.output_base + f'_train_{i}.mtz'
            ds.write_mtz(filename)

    if parser.embed:
        from matplotlib import pyplot as plt
        from IPython import embed
        embed(colors='Linux')


if __name__=="__main__":
    main()


#from careless.merge.merge import *
#import numpy as np
#
#def main(merger, parser, mlp_weights=None, freeze_mlp=False):
#    if parser.prior_mtz:
#        merger.append_reference_data(parser.prior_mtz)
#
#    if parser.separate_files:
#        merger.prep_indices(separate_files=True, image_id_key=parser.image_id_key)
#    else:
#        merger.prep_indices(separate_files=False, image_id_key=parser.image_id_key)
#
## Now populate merger.prior
#    if parser.normal_prior:
#        merger.add_normal_prior()
#    elif parser.rice_woolfson_prior:
#        merger.add_rice_woolfson_prior()
#    elif parser.laplace_prior:
#        merger.add_laplace_prior()
#    elif parser.studentt_prior_dof:
#        merger.add_studentt_prior(parser.studentt_prior_dof)
#    elif parser.wilson_prior_b:
#        merger.add_wilson_prior(b=parser.wilson_prior_b)
#    else:
#        merger.add_wilson_prior()
#
#    if parser.rice_woolfson_surrogate:
#        merger.add_rice_woolfson_posterior()
#    elif parser.folded_normal_surrogate:
#        merger.add_folded_normal_posterior()
#    else:
#        merger.add_truncated_normal_posterior()
#
##Now populate merger.likelihood
#    s = parser.mc_samples
#    if parser.laplace_likelihood:
#        merger.add_laplace_likelihood(parser.use_weights)
#    elif parser.studentt_likelihood_dof:
#        merger.add_studentt_likelihood(parser.studentt_likelihood_dof, parser.use_weights)
#    else:
#        merger.add_normal_likelihood(parser.use_weights)
#
#    metadata_keys = parser.metadata_keys.split(',')
#    positional_encoding_keys = parser.positional_encoding_keys
#    if positional_encoding_keys is not None:
#        positional_encoding_keys = positional_encoding_keys.split(',')
#
#    merger.add_scaling_model(
#        parser.sequential_layers, 
#        metadata_keys, 
#        width=parser.mlp_width, 
#        positional_encoding_frequencies=parser.positional_encoding_frequencies,
#        positional_encoding_keys=positional_encoding_keys
#    )
#    if mlp_weights is not None:
#        merger.scaling_model[0].set_weights(mlp_weights)
#    if freeze_mlp:
#        merger.scaling_model[0].trainable = False
#
#    if parser.image_scale_key is not None:
#        merger.add_image_scaler(parser.image_scale_key, parser.image_scale_prior)
#
#    losses = merger.train_model(
#        parser.iterations, 
#        mc_samples=parser.mc_samples, 
#        learning_rate=parser.learning_rate,
#        beta_1=parser.beta_1,
#        beta_2=parser.beta_2,
#        clip_value=parser.clip_value,
#        use_nadam=parser.use_nadam,
#    )
#    np.save(f"{parser.output_base}_losses.npy", losses)
#
#    for file_id,df in enumerate(merger.get_results()):
#        df.write_mtz(f"{parser.output_base}_{file_id}.mtz")
#
#    return merger
#
#
#if parser.type == 'mono':
#    merger = MonoMerger.from_mtzs(
#        *parser.mtzinput, 
#        anomalous=parser.anomalous, 
#        dmin=parser.dmin, 
#        isigi_cutoff=parser.isigi_cutoff, 
#        intensity_key=parser.intensity_key,
#        weight_kl=parser.multiplicity_weighted_elbo,
#    )
#    half1,half2 = MonoMerger.half_datasets_from_mtzs(
#        *parser.mtzinput, 
#        anomalous=parser.anomalous, 
#        seed=parser.seed,
#        dmin=parser.dmin, 
#        isigi_cutoff=parser.isigi_cutoff, 
#        intensity_key=parser.intensity_key,
#        weight_kl=parser.multiplicity_weighted_elbo,
#    )
#elif parser.type == 'poly':
#    merger = PolyMerger.from_mtzs(
#        *parser.mtzinput, 
#        anomalous=parser.anomalous, 
#        dmin=None, 
#        isigi_cutoff=parser.isigi_cutoff, 
#        intensity_key=parser.intensity_key,
#        weight_kl=parser.multiplicity_weighted_elbo,
#    )
#    half1,half2 = PolyMerger.half_datasets_from_mtzs(
#        *parser.mtzinput, 
#        anomalous=parser.anomalous, 
#        seed=parser.seed,
#        dmin=None, 
#        isigi_cutoff=parser.isigi_cutoff, 
#        intensity_key=parser.intensity_key,
#        weight_kl=parser.multiplicity_weighted_elbo,
#    )
#
#    merger.expand_harmonics(
#        dmin=parser.dmin, 
#        wavelength_key=parser.wavelength_key, 
#        wavelength_range=parser.wavelength_range
#    )
#    half1.expand_harmonics(
#        dmin=parser.dmin, 
#        wavelength_key=parser.wavelength_key, 
#        wavelength_range=parser.wavelength_range
#    )
#    half2.expand_harmonics(
#        dmin=parser.dmin, 
#        wavelength_key=parser.wavelength_key, 
#        wavelength_range=parser.wavelength_range
#    )
#else:
#    raise ValueError(f"Parser.type is {parser.type} but expected either 'mono' or 'poly'")
#
#merger = main(merger, parser)
#output_base = parser.output_base
#mlp_weights = merger.scaling_model[0].get_weights()
#if not parser.skip_xval:
#    parser.output_base = output_base + '_half1'
#    half1  = main(half1, parser, mlp_weights=mlp_weights, freeze_mlp=True)
#    parser.output_base = output_base + '_half2'
#    half2  = main(half2, parser, mlp_weights=mlp_weights, freeze_mlp=True)
#
#if parser.embed:
#    from matplotlib import pyplot as plt
#    from IPython import embed
#    embed(colors='Linux')
