import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import nibabel as nib
import numpy as np


def test_docker_inference_segmentation_simple(test_dir):
    """
    Testing the CLI within a Docker container for a simple segmentation inference unit test, running on CPU.
    The latest Docker image is being hosted at: dbouget/raidionics-segmenter:v1.4-py39-cpu

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference test in Docker container.\n")

    logging.info("Preparing configuration file.\n")
    try:
        image_name = "dbouget/raidionics-segmenter:v1.4-py39-cpu"
        if os.environ.get("GITHUB_ACTIONS"):
            image_name = "dbouget/raidionics-segmenter:" + os.environ["IMAGE_TAG"]


        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', '/workspace/resources/Inputs/PreopNeuro/inputs')
        seg_config.set('System', 'output_folder', '/workspace/resources/outputs')
        seg_config.set('System', 'model_folder', '/workspace/resources/Models/MRI_Brain')
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.set('Runtime', 'test_time_augmentation_iteration', '0')
        seg_config.set('Runtime', 'test_time_augmentation_fusion_mode', 'average')
        seg_config.set('Runtime', 'use_preprocessed_data', 'False')
        seg_config_filename = os.path.join(test_dir, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        logging.info("Running inference unit test in Docker container.\n")
        try:
            import platform
            cmd_docker = ['docker', 'run', '-v', '{}:/workspace/resources'.format(test_dir),
                          '--network=host', '--ipc=host']
            if not os.environ.get("GITHUB_ACTIONS") and sys.platform != "win32":
                cmd_docker.extend(['--user', str(os.geteuid())])
            cmd_docker.extend([image_name, '-c', '/workspace/resources/test_seg_config.ini', '-v', 'debug'])
            logging.info("Executing the following Docker call: {}".format(cmd_docker))
            if platform.system() == 'Windows':
                subprocess.check_call(cmd_docker, shell=True)
            else:
                subprocess.check_call(cmd_docker, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            raise ValueError("Error during inference test in Docker container.\n")

        logging.info("Collecting and comparing results.\n")
        segmentation_pred_filename = os.path.join(test_dir, "outputs", 'labels_Brain.nii.gz')
        assert os.path.exists(segmentation_pred_filename), "No brain mask was generated.\n"
        segmentation_gt_filename = os.path.join(test_dir, 'Inputs', 'PreopNeuro', 'inputs', 'input0_label_Brain.nii.gz')
        segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
        segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
        assert np.array_equal(segmentation_pred,
                              segmentation_gt), "Ground truth and prediction arrays are not identical"
    except Exception as e:
        logging.error(f"Error during inference Docker test with: \n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference Docker test.\n")