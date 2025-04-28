import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback


def test_inference_cli(data_test_classification_neuro):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard inference using the CLI.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(data_test_classification_neuro, "output_cli")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(data_test_classification_neuro, 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(data_test_classification_neuro, 'MRI_SequenceClassifier'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'probabilities')
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
            logging.error(f"Error during inference CLI test with: \n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference CLI test.\n")

        logging.info("Collecting and comparing results.\n")
        classification_results_filename = os.path.join(output_folder, 'classification-results.csv')
        assert os.path.exists(classification_results_filename), "Inference CLI test failed, no classification results were generated.\n"
    except Exception as e:
        logging.error(f"Error during inference CLI test with: \n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference CLI test.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test_inference_package(data_test_classification_neuro):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard inference test as a Python package.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(data_test_classification_neuro, "output_package")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        seg_config = configparser.ConfigParser()
        seg_config.add_section('System')
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(data_test_classification_neuro, 'inputs'))
        seg_config.set('System', 'output_folder', output_folder)
        seg_config.set('System', 'model_folder', os.path.join(data_test_classification_neuro, 'MRI_SequenceClassifier'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'folds_ensembling', 'False')
        seg_config.set('Runtime', 'ensembling_strategy', 'average')
        seg_config.set('Runtime', 'overlapping_ratio', '0.')
        seg_config.set('Runtime', 'reconstruction_method', 'probabilities')
        seg_config_filename = os.path.join(output_folder, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        try:
            logging.info("Running inference.\n")
            from raidionicsseg.fit import run_model
            run_model(seg_config_filename)

            logging.info("Collecting and comparing results.\n")
            classification_results_filename = os.path.join(output_folder, 'classification-results.csv')
            assert os.path.exists(classification_results_filename), "Inference CLI test failed, no classification results were generated.\n"
        except Exception as e:
            logging.error(f"Error during inference Python package test with: \n {traceback.format_exc()}.\n")
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise ValueError("Error during inference Python package test.\n")
    except Exception as e:
        logging.error(f"Error during inference Python package test with: \n {traceback.format_exc()}.\n")
        raise ValueError("Error during inference Python package test.\n")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)