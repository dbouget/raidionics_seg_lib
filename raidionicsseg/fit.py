import os
import sys
import time
import traceback
import logging
from raidionicsseg.Utils.configuration_parser import *
from raidionicsseg.PreProcessing.pre_processing import run_pre_processing
from raidionicsseg.Inference.predictions import run_predictions
from raidionicsseg.Inference.predictions_reconstruction import reconstruct_post_predictions
from raidionicsseg.Utils.io import dump_predictions, dump_classification_predictions
from raidionicsseg.Utils.configuration_parser import ConfigResources


def run_model(config_filename: str) -> None:
    """

    """
    ConfigResources.getInstance().init_environment(config_filename)
    # @TODO. Should have a check to know if segment or classify to run
    __segment()


def __segment() -> None:
    """

    """
    input_filename = ConfigResources.getInstance().input_volume_filename
    output_path = ConfigResources.getInstance().output_folder
    selected_model = ConfigResources.getInstance().model_folder

    logging.info("Starting inference for file: {}, with model: {}.\n".format(input_filename, selected_model))
    overall_start = start = time.time()
    pre_processing_parameters = ConfigResources.getInstance()
    model_path = os.path.join(selected_model, 'model.hd5')
    if not os.path.exists(model_path):
        raise ValueError('Requested model cannot be found on disk at location: \'{}\'.'.format(model_path))
    try:
        print('SLICERLOG: Preprocessing - Begin')
        nib_volume, resampled_volume, data, crop_bbox = run_pre_processing(filename=input_filename,
                                                                           pre_processing_parameters=pre_processing_parameters,
                                                                           storage_path=output_path)
        logging.info('Preprocessing: {} seconds.'.format(time.time() - start))
        print('SLICERLOG: Preprocessing - End')

        print('SLICERLOG: Inference - Begin')
        start = time.time()
        predictions = run_predictions(data=data, model_path=model_path, parameters=pre_processing_parameters)
        logging.info('Model loading + inference time: {} seconds.'.format(time.time() - start))
        print('SLICERLOG: Inference - End')

        print('SLICERLOG: Reconstruction - Begin')
        start = time.time()
        final_predictions = reconstruct_post_predictions(predictions=predictions, parameters=pre_processing_parameters,
                                                         crop_bbox=crop_bbox, nib_volume=nib_volume, resampled_volume=resampled_volume)
        logging.info('Prediction reconstruction time: {} seconds.'.format(time.time() - start))
        print('SLICERLOG: Reconstruction - End')

        print('SLICERLOG: Data dump - Begin')
        start = time.time()
        dump_predictions(predictions=final_predictions, parameters=pre_processing_parameters, nib_volume=nib_volume,
                         storage_path=output_path)
        logging.info('Data dump time: {} seconds.'.format(time.time() - start))
        print('SLICERLOG: Data dump - End')
        logging.info('Total processing time: {} seconds.\n'.format(time.time() - overall_start))
    except Exception as e:
        logging.error('{}\n'.format(traceback.format_exc()))


def __classify(input_filename, output_path, selected_model):
    """
    DEPRECATED -- TO UPDATE
    """
    print("Starting inference for file: {}, with model: {}.\n".format(input_filename, selected_model))
    overall_start = start = time.time()
    pre_processing_parameters = PreProcessingParser(model_name=selected_model)
    valid_extensions = ['.h5', '.hd5', '.hdf5', '.hdf', '.ckpt']
    model_path = ''
    for e, ext in enumerate(valid_extensions):
        model_path = os.path.join(MODELS_PATH, selected_model, 'model' + ext)
        if os.path.exists(model_path):
            break

    if not os.path.exists(model_path):
        raise ValueError('Could not find any model on Docker image matching the requested type \'{}\'.'.format(selected_model))

    print('SLICERLOG: Preprocessing - Begin')
    nib_volume, resampled_volume, data, crop_bbox = run_pre_processing(filename=input_filename,
                                                                       pre_processing_parameters=pre_processing_parameters,
                                                                       storage_prefix=output_path)
    print('Preprocessing: {} seconds.'.format(time.time() - start))
    print('SLICERLOG: Preprocessing - End')

    print('SLICERLOG: Inference - Begin')
    start = time.time()
    predictions = run_predictions(data=data, model_path=model_path, parameters=pre_processing_parameters)
    print('Model loading + inference time: {} seconds.'.format(time.time() - start))
    print('SLICERLOG: Inference - End')

    print('SLICERLOG: Data dump - Begin')
    dump_classification_predictions(predictions=predictions, parameters=pre_processing_parameters,
                                    storage_prefix=output_path)
    print('SLICERLOG: Data dump - End')
    print('Total processing time: {} seconds.\n'.format(time.time() - overall_start))

