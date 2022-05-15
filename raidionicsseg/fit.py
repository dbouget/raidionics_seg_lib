import os
import sys
import time
import traceback
import logging
import threading
import platform
import multiprocessing as mp
from .Utils.configuration_parser import *
from .PreProcessing.pre_processing import run_pre_processing
from .Inference.predictions import run_predictions
from .Inference.predictions_reconstruction import reconstruct_post_predictions
from .Utils.io import dump_predictions, dump_classification_predictions
from .Utils.configuration_parser import ConfigResources


def run_model_wrapper(config_filename: str) -> None:
    if platform.system() == 'Windows':
        from multiprocessing import freeze_support
        freeze_support()
    # run inference in a different process
    logging.debug("Spawning multiprocess...")
    mp.set_start_method('spawn', force=True)
    p = mp.Process(target=run_model, args=(config_filename,))
    p.start()
    p.join()
    logging.debug("Collecting results from multiprocess...")


def run_model(config_filename: str) -> None:
    """
    Entry point for running inference.
    All runtime parameters should be specified in a 'config.ini' file, according to the patterns indicated
    in 'blank_main_config.ini'.

    Parameters
    ----------
    config_filename: str
        Complete filepath indicating the configuration file to use for the inference.

    Returns
    -------
    None
    """
    config_parameters = ConfigResources()
    config_parameters.init_environment(config_filename)
    # @TODO. Maybe should store the segmentation/classification flag inside the model .ini
    if 'classifier' in os.path.basename(config_parameters.model_folder).lower():
        __classify(config_parameters)
    else:
        __segment(config_parameters)


def __segment(pre_processing_parameters: ConfigResources) -> None:
    """

    """
    input_filename = pre_processing_parameters.input_volume_filename
    output_path = pre_processing_parameters.output_folder
    selected_model = pre_processing_parameters.model_folder

    logging.info("Starting inference for file: {}, with model: {}.\n".format(input_filename, selected_model))
    overall_start = start = time.time()

    model_path = os.path.join(selected_model, 'model.hd5')
    if not os.path.exists(model_path):
        raise ValueError('Requested model cannot be found on disk at location: \'{}\'.\n'.format(model_path))
    try:
        logging.info('LOG: Preprocessing - Begin (1/4)\n')
        nib_volume, resampled_volume, data, crop_bbox = run_pre_processing(filename=input_filename,
                                                                           pre_processing_parameters=pre_processing_parameters,
                                                                           storage_path=output_path)
        logging.info('Preprocessing: {} seconds.\n'.format(time.time() - start))
        logging.info('LOG: Preprocessing - End (1/4)\n')

        logging.info('LOG: Inference - Begin (2/4)\n')
        start = time.time()
        predictions = run_predictions(data=data, model_path=model_path, parameters=pre_processing_parameters)
        logging.info('Model loading + inference time: {} seconds.\n'.format(time.time() - start))
        logging.info('LOG: Inference - End (2/4)\n')

        logging.info('LOG: Reconstruction - Begin (3/4)\n')
        start = time.time()
        final_predictions = reconstruct_post_predictions(predictions=predictions, parameters=pre_processing_parameters,
                                                         crop_bbox=crop_bbox, nib_volume=nib_volume, resampled_volume=resampled_volume)
        logging.info('Prediction reconstruction time: {} seconds.\n'.format(time.time() - start))
        logging.info('LOG: Reconstruction - End (3/4)\n')

        logging.info('LOG: Data dump - Begin (4/4)\n')
        start = time.time()
        dump_predictions(predictions=final_predictions, parameters=pre_processing_parameters, nib_volume=nib_volume,
                         storage_path=output_path)
        logging.info('Data dump time: {} seconds.\n'.format(time.time() - start))
        logging.info('LOG: Data dump - End (4/4)\n')
        logging.info('Total processing time: {} seconds.\n'.format(time.time() - overall_start))
    except Exception as e:
        logging.error('Segmentation failed to proceed with:\n {}\n'.format(traceback.format_exc()))
        raise RuntimeError("Segmentation failed to proceed.\n")


def __classify(pre_processing_parameters: ConfigResources):
    """

    """
    input_filename = pre_processing_parameters.input_volume_filename
    output_path = pre_processing_parameters.output_folder
    selected_model = pre_processing_parameters.model_folder

    logging.info("Starting inference for file: {}, with model: {}.\n".format(input_filename, selected_model))
    overall_start = start = time.time()

    model_path = os.path.join(selected_model, 'model.hd5')
    if not os.path.exists(model_path):
        raise ValueError('Requested model cannot be found on disk at location: \'{}\'.'.format(model_path))

    try:
        logging.info('LOG: Preprocessing - Begin (1/3)\n')
        nib_volume, resampled_volume, data, crop_bbox = run_pre_processing(filename=input_filename,
                                                                           pre_processing_parameters=pre_processing_parameters,
                                                                           storage_path=output_path)
        logging.info('Preprocessing: {} seconds.\n'.format(time.time() - start))
        logging.info('LOG: Preprocessing - End (1/3)\n')

        logging.info('LOG: Inference - Begin (2/3)\n')
        start = time.time()
        predictions = run_predictions(data=data, model_path=model_path, parameters=pre_processing_parameters)
        logging.info('Model loading + inference time: {} seconds.\n'.format(time.time() - start))
        logging.info('LOG: Inference - End (2/3)\n')

        logging.info('LOG: Data dump - Begin (3/3)\n')
        dump_classification_predictions(predictions=predictions, parameters=pre_processing_parameters,
                                        storage_path=output_path)
        logging.info('LOG: Data dump - End (3/3)\n')
        logging.info('Total processing time: {} seconds.\n'.format(time.time() - overall_start))
    except Exception as e:
        logging.error('Classification failed to process with:\n {}\n'.format(traceback.format_exc()))
        raise RuntimeError("Classification failed to proceed.\n")
