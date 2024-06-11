import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile

try:
    import requests
    import gdown
    if int(gdown.__version__.split('.')[0]) < 4 or int(gdown.__version__.split('.')[1]) < 4:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown==4.4.0'])
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'requests==2.28.2'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown==4.4.0'])
    import gdown
    import requests


def inference_test_torch_model():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running inference unit test.\n")
    logging.info("Downloading unit test resources.\n")
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    try:
        test_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Samples-RaidionicsSegLib-UnitTest3.zip'
        test_model_url = 'https://github.com/dbouget/Raidionics-models/releases/download/1.2.0/Raidionics-CT_Tumor-ONNX-v12.zip'

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

        archive_dl_dest = os.path.join(test_dir, 'tumor_model.zip')
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
        seg_config.set('System', 'gpu_id', "-1")
        seg_config.set('System', 'inputs_folder', os.path.join(test_dir, 'inputs'))
        seg_config.set('System', 'output_folder', test_dir)
        seg_config.set('System', 'model_folder', os.path.join(test_dir, 'CT_Tumor'))
        seg_config.add_section('Runtime')
        seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
        seg_config.set('Runtime', 'reconstruction_order', 'resample_first')
        seg_config.add_section('Mediastinum')
        seg_config.set('Mediastinum', 'lungs_segmentation_filename', os.path.join(test_dir, 'inputs',
                                                                                  'input0_label_lungs.nii.gz'))
        seg_config_filename = os.path.join(test_dir, 'test_seg_config.ini')
        with open(seg_config_filename, 'w') as outfile:
            seg_config.write(outfile)

        logging.info("Inference CLI unit test started.\n")
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
            logging.error("Error during inference CLI unit test with: \n {}.\n".format(traceback.format_exc()))
            shutil.rmtree(test_dir)
            raise ValueError("Error during inference CLI unit test.\n")

        logging.info("Collecting and comparing results.\n")
        tumor_segmentation_filename = os.path.join(test_dir, 'labels_Tumor.nii.gz')
        if not os.path.exists(tumor_segmentation_filename):
            logging.error("Inference CLI unit test failed, no tumor mask was generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Inference CLI unit test failed, no tumor mask was generated.\n")

        logging.info("Inference CLI unit test succeeded.\n")

        logging.info("Running inference.\n")
        from raidionicsseg.fit import run_model
        run_model(seg_config_filename)

        logging.info("Collecting and comparing results.\n")
        tumor_segmentation_filename = os.path.join(test_dir, 'labels_Tumor.nii.gz')
        if not os.path.exists(tumor_segmentation_filename):
            logging.error("Inference unit test failed, no tumor mask was generated.\n")
            raise ValueError("Inference unit test failed, no tumor mask was generated.\n")
        os.remove(tumor_segmentation_filename)
    except Exception as e:
        logging.error("Error during inference unit test with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during inference unit test.\n")

    logging.info("Inference unit test succeeded.\n")

    shutil.rmtree(test_dir)


if __name__ == "__main__":
    inference_test_torch_model()
