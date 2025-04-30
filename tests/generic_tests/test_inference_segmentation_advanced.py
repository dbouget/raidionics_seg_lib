import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback


def test_inference_segmentation_tta_single_input(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference with test-time augmentation.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_package_tta")
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
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.set('Runtime', 'test_time_augmentation_iteration', '2')
        seg_config.set('Runtime', 'test_time_augmentation_fusion_mode', 'average')
        seg_config_filename = os.path.join(output_folder, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        try:
            logging.info("Running inference\n")
            from raidionicsseg.fit import run_model
            run_model(seg_config_filename)

            logging.info("Collecting and comparing results.\n")
            brain_segmentation_filename = os.path.join(output_folder, 'labels_Brain.nii.gz')
            assert os.path.exists(brain_segmentation_filename), "No brain mask was generated.\n"
        except Exception as e:
            logging.error(f"Error during inference with TTA Python package test with: {e}\n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference with TTA Python package test.\n")
    except Exception as e:
        logging.error(f"Error during inference with TTA Python package test with: {e} \n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference with TTA Python package test.\n")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

def test_inference_segmentation_model_ensembling(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference with model ensembling.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_package_me")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(test_dir, 'Inputs', 'DiffLoader', 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(test_dir, 'Models', 'MRI_TumorCE_Postop/t1c_t1w_t1d'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', '3')
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
            logging.info("Running inference\n")
            from raidionicsseg.fit import run_model
            run_model(seg_config_filename)

            logging.info("Collecting and comparing results.\n")
            segmentation_pred_filename = os.path.join(output_folder, 'labels_TumorCE.nii.gz')
            assert os.path.exists(segmentation_pred_filename), "No segmentation mask was generated.\n"
        except Exception as e:
            logging.error(f"Error during model ensembling inference with: {e}\n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during model ensembling inference.\n")
    except Exception as e:
        logging.error(f"Error during model ensembling inference with: {e} \n {traceback.format_exc()}.\n")
        raise ValueError("Error during model ensembling inference.\n")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)