import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import nibabel as nib
import numpy as np


def test_inference_cli(test_dir, tmp_path):
    """
    Executing the module as a command line argument
    Parameters
    ----------
    input_data_dir
    input_models_dir

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard inference using the CLI.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_inference_cli")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        test_raw_input_fn = os.path.join(test_dir, "Inputs", 'PreopNeuro')
        tmp_test_input_fn = os.path.join(tmp_path, "results", "input_inference_cli")
        if os.path.exists(tmp_test_input_fn):
            shutil.rmtree(tmp_test_input_fn)
        shutil.copytree(test_raw_input_fn, tmp_test_input_fn)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(tmp_test_input_fn, 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(test_dir, 'Models', 'MRI_Brain'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.set('Runtime', 'test_time_augmentation_iteration', '0')
        seg_config.set('Runtime', 'test_time_augmentation_fusion_mode', 'average')
        seg_config_filename = os.path.join(output_folder, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        logging.info("Inference CLI test started.\n")
        try:
            import platform
            if platform.system() == 'Windows':
                subprocess.check_call(['raidionicsseg',
                                       '{config}'.format(config=seg_config_filename),
                                       '--verbose', 'debug'], shell=True)
            elif platform.system() == 'Darwin' and platform.processor() == 'arm':
                subprocess.check_call(['python3', '-m', 'raidionicsseg',
                                       '{config}'.format(config=seg_config_filename),
                                       '--verbose', 'debug'])
            else:
                subprocess.check_call(['raidionicsseg',
                                       '{config}'.format(config=seg_config_filename),
                                       '--verbose', 'debug'])
        except Exception as e:
            logging.error(f"Error during inference CLI test with: {e}\n {traceback.format_exc()}.\n")
            if os.path.exists(tmp_test_input_fn):
                shutil.rmtree(tmp_test_input_fn)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference CLI test.\n")

        logging.info("Collecting and comparing results.\n")
        segmentation_pred_filename = os.path.join(output_folder, 'labels_Brain.nii.gz')
        assert os.path.exists(segmentation_pred_filename), "No brain mask was generated.\n"
        segmentation_gt_filename = os.path.join(tmp_test_input_fn, 'inputs', 'input0_label_Brain.nii.gz')
        segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
        segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
        assert np.array_equal(segmentation_pred,
                              segmentation_gt), "Ground truth and prediction arrays are not identical"
    except Exception as e:
        logging.error(f"Error during inference CLI test with: {e}\n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference CLI test.\n")
    if os.path.exists(tmp_test_input_fn):
        shutil.rmtree(tmp_test_input_fn)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test_inference_package(test_dir, tmp_path):
    """
    Executing the module as a Python package
    Parameters
    ----------
    input_data_dir
    input_models_dir

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard inference test as a Python package.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_inference_package")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        test_raw_input_fn = os.path.join(test_dir, "Inputs", 'PreopNeuro')
        tmp_test_input_fn = os.path.join(tmp_path, "results", "input_inference_package")
        if os.path.exists(tmp_test_input_fn):
            shutil.rmtree(tmp_test_input_fn)
        shutil.copytree(test_raw_input_fn, tmp_test_input_fn)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(tmp_test_input_fn, 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(test_dir, 'Models', 'MRI_Brain'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.set('Runtime', 'test_time_augmentation_iteration', '0')
        seg_config.set('Runtime', 'test_time_augmentation_fusion_mode', 'average')
        seg_config_filename = os.path.join(output_folder, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        try:
            logging.info("Running inference.\n")
            from raidionicsseg.fit import run_model
            run_model(seg_config_filename)

            logging.info("Collecting and comparing results.\n")
            segmentation_pred_filename = os.path.join(output_folder, 'labels_Brain.nii.gz')
            assert os.path.exists(segmentation_pred_filename), "Inference CLI test failed, no brain mask was generated.\n"
            segmentation_gt_filename = os.path.join(tmp_test_input_fn, 'inputs', 'input0_label_Brain.nii.gz')
            segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
            segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
            assert np.array_equal(segmentation_pred, segmentation_gt), "Ground truth and prediction arrays are not identical"
        except Exception as e:
            logging.error(f"Error during inference Python package test with: {e} \n {traceback.format_exc()}.\n")
            if os.path.exists(tmp_test_input_fn):
                shutil.rmtree(tmp_test_input_fn)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference Python package test.\n")
    except Exception as e:
        logging.error(f"Error during inference Python package test with: {e}\n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference Python package test.\n")

    if os.path.exists(tmp_test_input_fn):
        shutil.rmtree(tmp_test_input_fn)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)