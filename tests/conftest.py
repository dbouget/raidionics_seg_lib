import os
import shutil
import pytest
import requests
import logging
import traceback
import zipfile
from tests.download_resources import download_resources


@pytest.fixture(scope="session")
def test_dir():
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.listdir(test_dir):
        download_resources(test_dir=test_dir)
    yield test_dir
    logging.info(f"Removing the temporary directory for tests.")
    shutil.rmtree(test_dir)
#
# @pytest.fixture(scope="session")
# def input_data_dir(temp_dir):
#     try:
#         test_preop_neuro_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsSegLib-UnitTest-PreopNeuro.zip'
#         test_diffloader_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsSegLib-UnitTest-DiffLoader.zip'
#         test_medi_url = "https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsSegLib-UnitTest-Mediastinum.zip"
#         dest_dir = os.path.join(temp_dir, "Inputs")
#         os.makedirs(dest_dir, exist_ok=True)
#
#         archive_dl_dest = os.path.join(dest_dir, 'preop_neuro_usecase.zip')
#         headers = {}
#         response = requests.get(test_preop_neuro_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'diffloader_usecase.zip')
#         headers = {}
#         response = requests.get(test_diffloader_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'mediastinum_usecase.zip')
#         headers = {}
#         response = requests.get(test_medi_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#     except Exception as e:
#         logging.error(f"Error during input data download with: {e} \n {traceback.format_exc()}\n")
#         shutil.rmtree(dest_dir)
#         raise ValueError("Error during input data download.\n")
#     return dest_dir
#
# @pytest.fixture(scope="session")
# def input_models_dir(temp_dir):
#     try:
#         brain_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_Brain-v13.zip'
#         postop_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_TumorCE_Postop-v13.zip'
#         seq_classif_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-MRI_SequenceClassifier-v13.zip'
#         ct_tumor_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Raidionics-CT_Tumor-v13.zip'
#         dest_dir = os.path.join(temp_dir, "Models")
#         os.makedirs(dest_dir, exist_ok=True)
#
#         archive_dl_dest = os.path.join(dest_dir, 'brain_model.zip')
#         headers = {}
#         response = requests.get(brain_model_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'postop_model.zip')
#         headers = {}
#         response = requests.get(postop_model_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'seq_classif_model.zip')
#         headers = {}
#         response = requests.get(seq_classif_model_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#         archive_dl_dest = os.path.join(dest_dir, 'ct_tumor_model.zip')
#         headers = {}
#         response = requests.get(ct_tumor_model_url, headers=headers, stream=True)
#         response.raise_for_status()
#         if response.status_code == requests.codes.ok:
#             with open(archive_dl_dest, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=1048576):
#                     f.write(chunk)
#         with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
#             zip_ref.extractall(dest_dir)
#
#     except Exception as e:
#         logging.error("Error during models download with: \n {}.\n".format(traceback.format_exc()))
#         shutil.rmtree(dest_dir)
#         raise ValueError("Error during models download.\n")
#     return dest_dir
