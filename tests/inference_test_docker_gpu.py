import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile

try:
    import requests
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'requests'])
    import requests


def inference_test_docker_gpu():
    """
    Testing the CLI within a Docker container for a simple segmentation inference unit test, running on GPU.
    A set of Docker image is available, based on the CUDA version running on the host machine:\n
    * dbouget/raidionics-segmenter:v1.2-py310-cuda-11.6

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference test in Docker container.\n")
    logging.info("Downloading unit test resources.\n")
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    try:
        test_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Samples-RaidionicsSegLib-UnitTest2.zip'
        test_model_url = 'https://github.com/dbouget/Raidionics-models/releases/download/1.2.0/Raidionics-MRI_Brain-ONNX-v12.zip'

        archive_dl_dest = os.path.join(test_dir, 'inference_volume.zip')
        headers = {}
        response = requests.get(test_image_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(test_dir)

        archive_dl_dest = os.path.join(test_dir, 'brain_model.zip')
        headers = {}
        response = requests.get(test_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(test_dir)

    except Exception as e:
        logging.error("Error during resources download with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during resources download.\n")

    logging.info("Preparing configuration file.\n")
    try:
        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "0")
        seg_config.set('System', 'inputs_folder', '/workspace/resources/inputs')
        seg_config.set('System', 'output_folder', '/workspace/resources/outputs')
        seg_config.set('System', 'model_folder', '/workspace/resources/MRI_Brain')
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.set('Runtime', 'use_preprocessed_data', 'False')
        seg_config_filename = os.path.join(test_dir, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        logging.info("Running inference unit test in Docker container.\n")
        try:
            import platform
            cmd_docker = ['docker', 'run', '-v', '{}:/workspace/resources'.format(test_dir), '--runtime=nvidia',
                          '--network=host', '--ipc=host', 'dbouget/raidionics-segmenter:v1.2-py310-cuda11.6',
                          '-c', '/workspace/resources/test_seg_config.ini', '-v', 'debug']
            logging.info("Executing the following Docker call: {}".format(cmd_docker))
            if platform.system() == 'Windows':
                subprocess.check_call(cmd_docker, shell=True)
            else:
                subprocess.check_call(cmd_docker)
        except Exception as e:
            logging.error("Error during inference test in Docker container with: \n {}.\n".format(traceback.format_exc()))
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            raise ValueError("Error during inference test in Docker container.\n")

        logging.info("Collecting and comparing results.\n")
        brain_segmentation_filename = os.path.join(test_dir, 'labels_Brain.nii.gz')
        if not os.path.exists(brain_segmentation_filename):
            logging.error("Inference in Docker container failed, no brain mask was generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Inference in Docker container failed, no brain mask was generated.\n")
        logging.info("Inference in Docker container succeeded.\n")
    except Exception as e:
        logging.error("Error during inference in Docker container with: \n {}.\n".format(traceback.format_exc()))
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        raise ValueError("Error during inference in Docker container with.\n")

    logging.info("Inference in Docker container succeeded.\n")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


inference_test_docker_gpu()
