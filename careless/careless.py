#!/usr/bin/env python


def main():
    from careless.parser import parser
    parser = parser.parse_args()
    run_careless(parser)

def run_careless(parser):
    # We defer all inputs to make sure the parser has priority in modifying tf parameters
    import tensorflow as tf
    import numpy as np
    import reciprocalspaceship as rs
    from careless.io.manager import DataManager
    from careless.io.formatter import MonoFormatter,LaueFormatter
    from careless.models.base import BaseModel
    from careless.models.merging.surrogate_posteriors import TruncatedNormal
    from careless.models.merging.variational import VariationalMergingModel
    from careless.models.scaling.image import HybridImageScaler,ImageScaler
    from careless.models.scaling.nn import MLPScaler

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
    if parser.use_image_scales:
        image_scaler = ImageScaler(n_images)
        scaler = HybridImageScaler(mlp_scaler, image_scaler)
    else:
        scaler = mlp_scaler

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

    hist = model.fit(train, epochs=parser.iterations, steps_per_epoch=1, verbose=0, callbacks=callbacks,  shuffle=False)

    for i,ds in enumerate(dm.get_results(surrogate_posterior, inputs=train)):
        filename = parser.output_base + f'_{i}.mtz'
        ds.write_mtz(filename)

    predictions_data = None
    if test is not None:
        for file_id, (ds_train, ds_test) in enumerate(zip(
                dm.get_predictions(model, train),
                dm.get_predictions(model, test),
                )):
            ds_train['test'] = rs.DataSeries(0, index=ds_train.index, dtype='I')
            ds_test['test']  = rs.DataSeries(1, index=ds_test.index, dtype='I')

            filename = parser.output_base + f'_predictions_{file_id}.mtz'
            ds_train.append(ds_test).write_mtz(filename)
    else:
        for file_id, ds_train in enumerate(dm.get_predictions(model, train)):
            ds_train['test'] = rs.DataSeries(0, index=ds_train.index, dtype='I')

            filename = parser.output_base + f'_predictions_{file_id}.mtz'
            ds_train.write_mtz(filename)

    if parser.merge_half_datasets:
        xval_data = [None] * len(dm.asu_collection)
        for repeat in range(parser.half_dataset_repeats):
            scaler.trainable = False
            for half_id, half in enumerate(dm.split_data_by_image()):
                surrogate_posterior = TruncatedNormal.from_loc_and_scale(loc, scale, low)
                model = VariationalMergingModel(surrogate_posterior, prior, likelihood, scaler, parser.mc_samples)

                opt = tf.keras.optimizers.Adam(
                    parser.learning_rate,
                    parser.beta_1,
                    parser.beta_2,
                )

                model.compile(opt)
                callbacks = [
                    ProgressBar(),
                ]

                model.fit(half, epochs=parser.iterations, steps_per_epoch=1, verbose=0, callbacks=callbacks,  shuffle=False)

                for file_id,ds in enumerate(dm.get_results(surrogate_posterior, inputs=half)):
                    ds['repeat'] = rs.DataSeries(repeat, index=ds.index, dtype='I')
                    ds['half'] = rs.DataSeries(half_id, index=ds.index, dtype='I')
                    if xval_data[file_id] is None:
                        xval_data[file_id] = ds
                    else:
                        xval_data[file_id] = xval_data[file_id].append(ds)

        for file_id, ds in enumerate(xval_data):
            filename = parser.output_base + f'_xval_{file_id}.mtz'
            ds.write_mtz(filename)

    if parser.embed:
        from IPython import embed
        embed(colors='Linux')


if __name__=="__main__":
    main()

