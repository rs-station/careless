#!/usr/bin/env python


def main():
    from . import __version__
    print(f"Careless version {__version__}")
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
    elif parser.type == 'mono':
        df = MonoFormatter.from_parser(parser)


    inputs,rac = df.format_files(parser.reflection_files)
    dm = DataManager(inputs, rac, parser=parser)

    if parser.test_fraction is not None:
        train,test = dm.split_data_by_refl(parser.test_fraction)
    else:
        train,test = dm.inputs,None

    model = dm.build_model()

    if parser.scale_file is not None:
        model.scaling_model.load_weights(parser.scale_file)
    if parser.freeze_scales:
        model.scaling_model.trainable = False

    if parser.structure_factor_file is not None:
        model.surrogate_posterior.load_weights(parser.structure_factor_file)
    if parser.freeze_structure_factors:
        model.surrogate_posterior.trainable = False

    validation_frequency = parser.validation_frequency
    progress = not parser.disable_progress_bar

    history = model.train_model(
        tuple(map(tf.convert_to_tensor, train)),
        parser.iterations,
        message="Training",
        validation_data=test,
        validation_frequency=validation_frequency,
        progress=progress,
    )

    for i,ds in enumerate(dm.get_results(model.surrogate_posterior, inputs=train)):
        filename = parser.output_base + f'_{i}.mtz'
        ds.write_mtz(filename)

    filename = parser.output_base + f'_history.csv'
    history = rs.DataSet(history).to_csv(filename, index_label='step')

    model.surrogate_posterior.save_weights(parser.output_base + '_structure_factor')
    model.scaling_model.save_weights(parser.output_base + '_scale')
    import pickle
    with open(parser.output_base + "_data_manager.pickle", "wb") as out:
        pickle.dump(dm, out)

    predictions_data = None
    if test is not None:
        for file_id, (ds_train, ds_test) in enumerate(zip(
                dm.get_predictions(model, train),
                dm.get_predictions(model, test),
                )):
            ds_train['test'] = rs.DataSeries(0, index=ds_train.index, dtype='I')
            ds_test['test']  = rs.DataSeries(1, index=ds_test.index, dtype='I')

            filename = parser.output_base + f'_predictions_{file_id}.mtz'
            rs.concat((
                ds_train,
                ds_test,
            )).write_mtz(filename)
    else:
        for file_id, ds_train in enumerate(dm.get_predictions(model, train)):
            ds_train['test'] = rs.DataSeries(0, index=ds_train.index, dtype='I')

            filename = parser.output_base + f'_predictions_{file_id}.mtz'
            ds_train.write_mtz(filename)

    if parser.merge_half_datasets:
        scaling_model = model.scaling_model
        scaling_model.trainable = False
        xval_data = [None] * len(dm.asu_collection)
        for repeat in range(parser.half_dataset_repeats):
            for half_id, half in enumerate(dm.split_data_by_image()):
                model = dm.build_model(scaling_model=scaling_model)
                history = model.train_model(
                    tuple(map(tf.convert_to_tensor, half)), 
                    parser.iterations,
                    message=f"Merging repeat {repeat+1} half {half_id+1}",
                    progress=progress,
                )

                for file_id,ds in enumerate(dm.get_results(model.surrogate_posterior, inputs=half)):
                    ds['repeat'] = rs.DataSeries(repeat, index=ds.index, dtype='I')
                    ds['half'] = rs.DataSeries(half_id, index=ds.index, dtype='I')
                    if xval_data[file_id] is None:
                        xval_data[file_id] = ds
                    else:
                        xval_data[file_id] = rs.concat((xval_data[file_id], ds))

        for file_id, ds in enumerate(xval_data):
            filename = parser.output_base + f'_xval_{file_id}.mtz'
            ds.write_mtz(filename)

    if parser.embed:
        from IPython import embed
        embed(colors='Linux')


if __name__=="__main__":
    main()

