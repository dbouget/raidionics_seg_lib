import os
import shutil
import pytest
import requests
import logging
import traceback
import zipfile


@pytest.fixture(scope="session")
def temp_dir():
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    yield test_dir
    logging.info(f"Removing the temporary directory for tests.")
    shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def data_test2(temp_dir):
    try:
        test_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Samples-RaidionicsSegLib-UnitTest2.zip'
        test_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_Brain-v13.zip'
        dest_dir = os.path.join(temp_dir, "Test2")
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'inference_volume.zip')
        headers = {}
        response = requests.get(test_image_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'brain_model.zip')
        headers = {}
        response = requests.get(test_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    except Exception as e:
        logging.error("Error during resources download with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(dest_dir)
        raise ValueError("Error during resources download.\n")
    return dest_dir


@pytest.fixture(scope="session")
def data_test3(temp_dir):
    try:
        test_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Samples-RaidionicsSegLib-UnitTest3.zip'
        test_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-CT_Tumor-v13.zip'
        dest_dir = os.path.join(temp_dir, "Test3")
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'inference_volume.zip')
        headers = {}
        response = requests.get(test_image_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        archive_dl_dest = os.path.join(dest_dir, 'tumor_model.zip')
        headers = {}
        response = requests.get(test_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

    except Exception as e:
        logging.error("Error during resources download with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(dest_dir)
        raise ValueError("Error during resources download.\n")
    return dest_dir