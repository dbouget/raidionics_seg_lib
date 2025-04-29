import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile


def test_docker_inference_segmentation_simple(data_test2):
    """
    Testing the CLI within a Docker container for a simple segmentation inference unit test, running on GPU.
    The latest Docker image is being hosted at: dbouget/raidionics-segmenter:v1.4-py39-gpu

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference test in Docker container.\n")

    logging.info("Preparing configuration file.\n")
    try:
        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "0")
        seg_config.set('System', 'inputs_folder', '/workspace/resources/inputs')
        seg_config.set('System', 'output_folder', '/workspace/resources/outputs')
        seg_config.set('System', 'model_folder', '/workspace/resources/MRI_Brain')
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.set('Runtime', 'test_time_augmentation_iteration', '0')
        seg_config.set('Runtime', 'test_time_augmentation_fusion_mode', 'average')
        seg_config.set('Runtime', 'use_preprocessed_data', 'False')
        seg_config_filename = os.path.join(data_test2, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        logging.info("Running inference unit test in Docker container.\n")
        try:
            import platform
            cmd_docker = ['docker', 'run', '-v', '{}:/workspace/resources'.format(data_test2),
                          '--network=host', '--ipc=host', '--gpus=all', '--user', str(os.geteuid()),
                          'dbouget/raidionics-segmenter:v1.4-py39-gpu',
                          '-c', '/workspace/resources/test_seg_config.ini', '-v', 'debug']
            logging.info("Executing the following Docker call: {}".format(cmd_docker))
            if platform.system() == 'Windows':
                subprocess.check_call(cmd_docker, shell=True)
            else:
                subprocess.check_call(cmd_docker)
        except Exception as e:
            raise ValueError("Error during inference test in Docker container.\n")

        logging.info("Collecting and comparing results.\n")
        brain_segmentation_filename = os.path.join(data_test2, "outputs", 'labels_Brain.nii.gz')
        assert os.path.exists(
            brain_segmentation_filename), "Inference Docker test failed, no brain mask was generated.\n"
    except Exception as e:
        logging.error(f"Error during inference Docker test with: \n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference Docker test.\n")