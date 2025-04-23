import os
import glob
import sys
import time
import traceback
import logging
import threading
import platform
import multiprocessing as mp
from .Utils.configuration_parser import *
from .PreProcessing.pre_processing import prepare_pre_processing
from .Inference.predictions import run_predictions
from .Inference.predictions_classification import run_predictions_classification
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


def run_model(config_filename: str, logging_filename: str = None) -> None:
    """
    Entry point for running inference.
    All runtime parameters should be specified in a 'config.ini' file, according to the patterns indicated
    in 'blank_main_config.ini'.

    Parameters
    ----------
    config_filename: str
        Complete filepath indicating the configuration file to use for the inference.
    logging_filename: str
        Complete filepath to a log file on disk, where the logging output will be written.
    Returns
    -------
    None
    """
    if logging_filename:
        logger = logging.getLogger()
        handler = logging.FileHandler(filename=logging_filename, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter(fmt="%(asctime)s ; %(name)s ; %(levelname)s ; %(message)s",
                                               datefmt='%d/%m/%Y %H.%M'))
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    config_parameters = ConfigResources()
    config_parameters.init_environment(config_filename)
    # @TODO. Maybe should store the segmentation/classification flag inside the model .ini
    if config_parameters.task == "classification":
        __classify(config_parameters)
    else:
        __segment(config_parameters)


def __segment(pre_processing_parameters: ConfigResources) -> None:
    """

    """
    inputs_folder = pre_processing_parameters.inputs_folder
    output_path = pre_processing_parameters.output_folder
    base_model_path = pre_processing_parameters.model_folder

    if not os.path.exists(base_model_path):
        raise ValueError('Requested model cannot be found on disk at location: \'{}\'.'.format(base_model_path))

    logging.info("Starting inference for folder: {}, with model: {}.".format(os.path.basename(inputs_folder),
                                                                             os.path.basename(base_model_path)))
    # models_path = sorted(glob.glob(os.path.join(base_model_path, "**", "*")), key=lambda x: x.split('/')[-2][0])
    models_path = sorted(glob.glob(os.path.join(base_model_path, "**", "*")),
                         key=lambda x: os.path.basename(os.path.dirname(x)))
    if not pre_processing_parameters.predictions_folds_ensembling:
        models_path = [models_path[0]]
    logging.info('LOG: Segmentation - 4 steps.')
    overall_start = start = time.time()

    try:
        logging.info('LOG: Segmentation - Preprocessing - Begin (1/4)')
        nib_volume, resampled_volume, data, crop_bbox = prepare_pre_processing(folder=inputs_folder,
                                                                               pre_processing_parameters=pre_processing_parameters,
                                                                               storage_path=output_path)
        logging.info('LOG: Segmentation - Runtime: {} seconds.'.format(time.time() - start))
        logging.info('LOG: Segmentation - Preprocessing - End (1/4)')

        logging.info('LOG: Segmentation - Inference - Begin (2/4)')
        start = time.time()
        predictions = run_predictions(data=data, models_path=models_path, parameters=pre_processing_parameters)
        logging.info('LOG: Segmentation - Runtime: {} seconds.'.format(time.time() - start))
        logging.info('LOG: Segmentation - Inference - End (2/4)')

        logging.info('LOG: Segmentation - Reconstruction - Begin (3/4)')
        start = time.time()
        final_predictions = reconstruct_post_predictions(predictions=predictions, parameters=pre_processing_parameters,
                                                         crop_bbox=crop_bbox, nib_volume=nib_volume,
                                                         resampled_volume=resampled_volume)
        logging.info('LOG: Segmentation - Runtime: {} seconds.'.format(time.time() - start))
        logging.info('LOG: Segmentation - Reconstruction - End (3/4)')

        logging.info('LOG: Segmentation - Data dump - Begin (4/4)')
        start = time.time()
        dump_predictions(predictions=final_predictions, parameters=pre_processing_parameters, nib_volume=nib_volume,
                         storage_path=output_path)
        logging.info('LOG: Segmentation - Runtime: {} seconds.'.format(time.time() - start))
        logging.info('LOG: Segmentation - Data dump - End (4/4)')

        logging.info('Total processing time: {} seconds.'.format(time.time() - overall_start))
    except Exception as e:
        logging.error('Segmentation failed to proceed with:\n {}'.format(traceback.format_exc()))
        raise RuntimeError("Segmentation failed to proceed.")


def __classify(pre_processing_parameters: ConfigResources):
    """

    """
    inputs_folder = pre_processing_parameters.inputs_folder
    output_path = pre_processing_parameters.output_folder
    base_model_path = pre_processing_parameters.model_folder

    if not os.path.exists(base_model_path):
        raise ValueError('Requested model cannot be found on disk at location: \'{}\'.'.format(base_model_path))

    logging.info("Starting inference for folder: {}, with model: {}.".format(os.path.basename(inputs_folder),
                                                                             base_model_path))
    logging.info('LOG: Classification - 3 steps.')
    overall_start = start = time.time()

    #models_path = sorted(glob.glob(os.path.join(base_model_path, "**", "*")), key=lambda x: x.split('/')[-2][0])
    models_path = sorted(glob.glob(os.path.join(base_model_path, "**", "*")),
                         key=lambda x: os.path.basename(os.path.dirname(x)))
    if not pre_processing_parameters.predictions_folds_ensembling:
        models_path = [models_path[0]]

    try:
        logging.info('LOG: Classification - Preprocessing - Begin (1/3)')
        nib_volume, resampled_volume, data, crop_bbox = prepare_pre_processing(folder=inputs_folder,
                                                                               pre_processing_parameters=pre_processing_parameters,
                                                                               storage_path=output_path)
        logging.info('LOG: Classification - Runtime: {} seconds.'.format(time.time() - start))
        logging.info('LOG: Classification - Preprocessing - End (1/3)')

        logging.info('LOG: Classification - Inference - Begin (2/3)')
        start = time.time()
        predictions = run_predictions_classification(data=data, models_path=models_path, parameters=pre_processing_parameters)
        logging.info('LOG: Classification - Runtime: {} seconds.'.format(time.time() - start))
        logging.info('LOG: Classification - Inference - End (2/3)')

        logging.info('LOG: Classification - Data dump - Begin (3/3)')
        dump_classification_predictions(predictions=predictions, parameters=pre_processing_parameters,
                                        storage_path=output_path)
        logging.info('LOG: Classification - Runtime: {} seconds.'.format(time.time() - start))
        logging.info('LOG: Classification - Data dump - End (3/3)')
        logging.info('Total processing time: {} seconds.'.format(time.time() - overall_start))
    except Exception as e:
        logging.error('Classification failed to process with:\n {}'.format(traceback.format_exc()))
        raise RuntimeError("Classification failed to proceed.")
