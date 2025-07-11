import os
import shutil
import configparser
import logging
import traceback
import nibabel as nib
import numpy as np


def test_inference_segmentation_tta_single_input(test_dir, tmp_path):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference with test-time augmentation.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_package_tta_single")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        test_raw_input_fn = os.path.join(test_dir, "Inputs", 'PreopNeuro')
        tmp_test_input_fn = os.path.join(tmp_path, "results", "input_package_tta_single")
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
            segmentation_pred_filename = os.path.join(output_folder, 'labels_Brain.nii.gz')
            assert os.path.exists(segmentation_pred_filename), "No brain mask was generated.\n"
            segmentation_gt_filename = os.path.join(tmp_test_input_fn, 'inputs', 'input0_label_Brain.nii.gz')
            segmentation_pred_nib = nib.load(segmentation_pred_filename)
            segmentation_gt_nib = nib.load(segmentation_gt_filename)
            pred_volume = np.count_nonzero(segmentation_pred_nib.get_fdata()[:]) * np.prod(
                segmentation_pred_nib.header.get_zooms()[0:3]) * 1e-3
            gt_volume = np.count_nonzero(segmentation_gt_nib.get_fdata()[:]) * np.prod(
                segmentation_gt_nib.header.get_zooms()[0:3]) * 1e-3
            logging.info(f"Volume difference: {abs(pred_volume - gt_volume)}\n")
            assert abs(pred_volume - gt_volume) < 0.5, \
                "Ground truth and prediction arrays are very different"
        except Exception as e:
            logging.error(f"Error during inference with TTA Python package test with: {e}\n {traceback.format_exc()}.\n")
            if os.path.exists(tmp_test_input_fn):
                shutil.rmtree(tmp_test_input_fn)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference with TTA Python package test.\n")
    except Exception as e:
        logging.error(f"Error during inference with TTA Python package test with: {e} \n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference with TTA Python package test.\n")

    if os.path.exists(tmp_test_input_fn):
        shutil.rmtree(tmp_test_input_fn)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

def test_inference_segmentation_model_ensembling(test_dir, tmp_path):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference with model ensembling.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_package_me")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        test_raw_input_fn = os.path.join(test_dir, "Inputs", 'DiffLoader')
        tmp_test_input_fn = os.path.join(tmp_path, "results", "input_package_me")
        if os.path.exists(tmp_test_input_fn):
            shutil.rmtree(tmp_test_input_fn)
        shutil.copytree(test_raw_input_fn, tmp_test_input_fn)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(tmp_test_input_fn, 'inputs'))
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
            segmentation_gt_filename = os.path.join(tmp_test_input_fn, 'verif', 'input0_labels_TumorCE_foldensemble.nii.gz')
            segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
            segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
            logging.info(
                f"Ground truth and prediction arrays difference: {np.count_nonzero(abs(segmentation_gt - segmentation_pred))} pixels")
            assert np.array_equal(segmentation_pred, segmentation_gt), "Ground truth and prediction arrays are not identical"
            # assert np.count_nonzero(np.abs(
            #     segmentation_pred - segmentation_gt)) < 200, "Ground truth and prediction arrays are very different"
        except Exception as e:
            logging.error(f"Error during model ensembling inference with: {e}\n {traceback.format_exc()}.\n")
            if os.path.exists(tmp_test_input_fn):
                shutil.rmtree(tmp_test_input_fn)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during model ensembling inference.\n")
    except Exception as e:
        logging.error(f"Error during model ensembling inference with: {e} \n {traceback.format_exc()}.\n")
        raise ValueError("Error during model ensembling inference.\n")

    if os.path.exists(tmp_test_input_fn):
        shutil.rmtree(tmp_test_input_fn)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)