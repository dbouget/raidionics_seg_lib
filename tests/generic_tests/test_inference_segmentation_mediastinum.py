import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback


def test_inference_cli(data_test3):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard inference using the CLI for a mediastinum model.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(data_test3, "output_cli")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(data_test3, 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(data_test3, 'CT_Tumor'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.set('Runtime', 'test_time_augmentation_iteration', '0')
        seg_config.set('Runtime', 'test_time_augmentation_fusion_mode', 'average')
        seg_config.add_section('Mediastinum')
        seg_config.set('Mediastinum', 'lungs_segmentation_filename', os.path.join(data_test3, 'inputs',
                                                                                  'input0_label_lungs.nii.gz'))
        seg_config_filename = os.path.join(output_folder, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        logging.info("Starting inference.\n")
        try:
            import platform
            if platform.system() == 'Windows':
                subprocess.check_call(['raidionicsseg',
                                       '{config}'.format(config=seg_config_filename),
                                       '--verbose', 'debug'], shell=True)
            elif platform.system() == 'Darwin' and platform.processor() == 'arm':
                # subprocess.check_call(['python3', '-m', 'raidionicsseg',
                #                        '{config}'.format(config=seg_config_filename),
                #                        '--verbose', 'debug'])
                result = subprocess.run(['python3', '-m', 'raidionicsseg',
                     '{config}'.format(config=seg_config_filename),
                     '--verbose', 'debug'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                logging.info("STDOUT:", result.stdout)
                logging.error("STDERR:", result.stderr)
                result.check_returncode()
            else:
                subprocess.check_call(['raidionicsseg',
                                       '{config}'.format(config=seg_config_filename),
                                       '--verbose', 'debug'])
        except Exception as e:
            logging.error(f"Error during test with: \n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during test.\n")

        logging.info("Collecting and comparing results.\n")
        segmentation_filename = os.path.join(output_folder, 'labels_Tumor.nii.gz')
        assert os.path.exists(segmentation_filename), "Inference CLI test failed, no tumor mask was generated.\n"
    except Exception as e:
        logging.error(f"Error during test with: \n {traceback.format_exc()}.\n")
        raise ValueError("Error during test.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test_inference_package(data_test3):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard inference test as a Python package for a mediastinum model.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(data_test3, "output_package")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(data_test3, 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(data_test3, 'CT_Tumor'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.set('Runtime', 'test_time_augmentation_iteration', '0')
        seg_config.set('Runtime', 'test_time_augmentation_fusion_mode', 'average')
        seg_config.add_section('Mediastinum')
        seg_config.set('Mediastinum', 'lungs_segmentation_filename', os.path.join(data_test3, 'inputs',
                                                                                  'input0_label_lungs.nii.gz'))
        seg_config_filename = os.path.join(output_folder, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        try:
            logging.info("Starting inference.\n")
            from raidionicsseg.fit import run_model
            run_model(seg_config_filename)

            logging.info("Collecting and comparing results.\n")
            segmentation_filename = os.path.join(output_folder, 'labels_Tumor.nii.gz')
            assert os.path.exists(segmentation_filename), "Inference CLI test failed, no tumor mask was generated.\n"
        except Exception as e:
            logging.error(f"Error during test with: \n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during test.\n")
    except Exception as e:
        logging.error(f"Error during test with: \n {traceback.format_exc()}.\n")
        raise ValueError("Error during test.\n")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

