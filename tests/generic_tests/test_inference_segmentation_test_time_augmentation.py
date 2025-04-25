import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback


def test_inference_cli_tta(data_test2):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running segmentation inference with TTA using the CLI.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(data_test2, "output_cli_tta")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(data_test2, 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(data_test2, 'MRI_Brain'))
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

        logging.info("Inference with TTA CLI test started.\n")
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
            logging.error(f"Error during inference with TTA CLI test with: \n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference with TTA CLI test.\n")

        logging.info("Collecting and comparing results.\n")
        brain_segmentation_filename = os.path.join(output_folder, 'labels_Brain.nii.gz')
        assert os.path.exists(brain_segmentation_filename), "Inference with TTA CLI test failed, no brain mask was generated.\n"
    except Exception as e:
        logging.error(f"Error during inference with TTA CLI test with: \n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference with TTA CLI test.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test_inference_package_tta(data_test2):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference with TTA test as a Python package.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(data_test2, "output_package_tta")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(data_test2, 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(data_test2, 'MRI_Brain'))
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
            logging.info("Running inference with TTA.\n")
            from raidionicsseg.fit import run_model
            run_model(seg_config_filename)

            logging.info("Collecting and comparing results.\n")
            brain_segmentation_filename = os.path.join(output_folder, 'labels_Brain.nii.gz')
            assert os.path.exists(brain_segmentation_filename), "Inference with TTA Python test failed, no brain mask was generated.\n"
        except Exception as e:
            logging.error(f"Error during inference with TTA Python package test with: \n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference with TTA Python package test.\n")
    except Exception as e:
        logging.error(f"Error during inference with TTA Python package test with: \n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference with TTA Python package test.\n")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)