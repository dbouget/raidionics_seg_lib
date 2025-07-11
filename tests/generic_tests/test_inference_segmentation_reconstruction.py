import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import nibabel as nib
import numpy as np


def test_inference_segmentation_reconstruction_order(test_dir):
    """
    Executing the module as a Python package
    Parameters
    ----------
    test_dir

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference segmentation reconstruction test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_package")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(test_dir, 'Inputs', 'PreopNeuro', 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(test_dir, 'Models', 'MRI_Brain'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_second')
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
            segmentation_gt_filename = os.path.join(test_dir, 'Inputs', 'PreopNeuro', 'verif', 'input0_labels_Brain_resample_second.nii.gz')
            segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
            segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
            # assert np.array_equal(segmentation_pred, segmentation_gt), "Ground truth and prediction arrays are not identical"
        except Exception as e:
            logging.error(f"Error during inference Python package test with: {e} \n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference Python package test.\n")
    except Exception as e:
        logging.error(f"Error during inference Python package test with: {e}\n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference Python package test.\n")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test_inference_segmentation_reconstruction_method(test_dir):
    """
    Executing the module as a Python package
    Parameters
    ----------
    test_dir

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference segmentation reconstruction test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_package")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(test_dir, 'Inputs', 'PreopNeuro', 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(test_dir, 'Models', 'MRI_Brain'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'probabilities')
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
            segmentation_pred_filename = os.path.join(output_folder, 'pred_Brain.nii.gz')
            assert os.path.exists(segmentation_pred_filename), "Inference CLI test failed, no brain mask was generated.\n"
            segmentation_gt_filename = os.path.join(test_dir, 'Inputs', 'PreopNeuro', 'verif', 'input0_pred_Brain.nii.gz')
            logging.info(f"Comparing {segmentation_pred_filename} with original {segmentation_gt_filename}")
            segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
            segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
            logging.info(
                f"Ground truth and prediction arrays difference: {np.count_nonzero(abs(segmentation_gt - segmentation_pred))} pixels")
            # assert np.array_equal(segmentation_pred, segmentation_gt), "Ground truth and prediction arrays are not identical"
        except Exception as e:
            logging.error(f"Error during inference Python package test with: {e} \n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference Python package test.\n")
    except Exception as e:
        logging.error(f"Error during inference Python package test with: {e}\n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference Python package test.\n")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)