import configparser
from typing import Tuple, List
import numpy as np
from copy import deepcopy
from scipy.ndimage import binary_fill_holes
from nibabel.processing import resample_to_output
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
import subprocess
import shutil
import os
from raidionicsseg.Utils.io import load_nifti_volume, convert_and_export_to_nifti
from raidionicsseg.Utils.configuration_parser import generate_runtime_config
from raidionicsseg.Utils.configuration_parser import ConfigResources


def crop_MR_background(filepath: str, volume: np.ndarray, new_spacing: Tuple[float], storage_path: str,
                       parameters: ConfigResources) -> Tuple[np.ndarray, List[int]]:
    """
    Performs different background cropping inside an MRI volume, as defined by the 'crop_background' stored in a model
    preprocessing configuration file.
    In 'minimum' mode, the black space around the head is removed, in 'brain_mask' mode all voxels not belonging to the
    brain are set to rgb(0, 0, 0), and in 'brain_clip' mode only the smallest bounding region around the brain is kept.

    Parameters
    ----------
    filepath : str
        Filepath of the input volume (CT or MRI) to use.
    volume : np.ndarray
        .
    new_spacing : Tuple[float]
        .
    storage_path : str
        .
    parameters : :obj:`ConfigResources`
        .
    Returns
    -------
    np.ndarray
        New volume after background cropping procedure.
    List[int]
        Indices of a bounding region within the volume for additional cropping (e.g. coordinates around the head,
        or tightly around the brain only).
        The bounding region is expressed as: [minx, miny, minz, maxx, maxy, maxz].
    """
    if parameters.crop_background == 'minimum':
        return crop_MR(volume, parameters)
    elif parameters.crop_background == 'brain_clip' or parameters.crop_background == 'brain_mask':
        return brain_selection_DL(filepath, volume, new_spacing, storage_path, parameters)


def crop_MR(volume, parameters):
    original_volume = np.copy(volume)
    volume[volume >= 0.2] = 1
    volume[volume < 0.2] = 0
    volume = volume.astype(np.uint8)
    volume = binary_fill_holes(volume).astype(np.uint8)
    regions = regionprops(volume)
    min_row, min_col, min_depth, max_row, max_col, max_depth = regions[0].bbox
    print('cropping params', min_row, min_col, min_depth, max_row, max_col, max_depth)

    cropped_volume = original_volume[min_row:max_row, min_col:max_col, min_depth:max_depth]
    bbox = [min_row, min_col, min_depth, max_row, max_col, max_depth]

    return cropped_volume, bbox


def brain_selection_DL(filepath, volume, new_spacing, storage_path, parameters):
    if not os.path.exists(parameters.runtime_brain_mask_filepath):
        brain_config_filename = os.path.join(os.path.dirname(parameters.config_filename), 'brain_main_config.ini')
        new_parameters = configparser.ConfigParser()
        new_parameters.read(parameters.config_filename)
        new_parameters.set('System', 'model_folder', os.path.join(os.path.dirname(parameters.model_folder), 'MRI_Brain'))
        new_parameters.set('Runtime', 'reconstruction_method', 'thresholding')
        new_parameters.set('Runtime', 'reconstruction_order', 'resample_first')
        with open(brain_config_filename, 'w') as cf:
            new_parameters.write(cf)
        # script_path = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-2]) + '/main.py'
        script_path = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]) + '/__main__.py'
        subprocess.call(['python3', '{script}'.format(script=script_path),
                         '{config}'.format(config=brain_config_filename)])
        brain_mask_filename = os.path.join(storage_path, 'labels_Brain.nii.gz')
        os.remove(brain_config_filename)
    else:
        brain_mask_filename = parameters.runtime_brain_mask_filepath

    brain_mask_ni = load_nifti_volume(brain_mask_filename)
    resampled_volume = resample_to_output(brain_mask_ni, new_spacing, order=0)
    brain_mask = resampled_volume.get_data().astype('uint8')
    # brain_mask = resampled_volume.get_data().astype('float32')
    # brain_mask[brain_mask < 0.5] = 0
    # brain_mask[brain_mask >= 0.5] = 1
    # brain_mask = brain_mask.astype('uint8')

    labels, nb_components = label(brain_mask)
    brain_objects_properties = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)

    brain_object = brain_objects_properties[0]
    brain_component = np.zeros(brain_mask.shape).astype('uint8')
    brain_component[brain_object.bbox[0]:brain_object.bbox[3],
    brain_object.bbox[1]:brain_object.bbox[4],
    brain_object.bbox[2]:brain_object.bbox[5]] = 1

    final_brain_mask = brain_mask & brain_component

    return advanced_crop_exclude_background(volume, crop_mode=parameters.crop_background, brain_mask=final_brain_mask)


def advanced_crop_exclude_background(data, crop_mode, brain_mask):
    original_data = np.copy(data)
    regions = regionprops(brain_mask)
    min_row, min_col, min_depth, max_row, max_col, max_depth = regions[0].bbox
    print('cropping params', min_row, min_col, min_depth, max_row, max_col, max_depth)

    if crop_mode == 'brain_mask':
        original_data[brain_mask == 0] = 0
    cropped_data = original_data[min_row:max_row, min_col:max_col, min_depth:max_depth]
    bbox = [min_row, min_col, min_depth, max_row, max_col, max_depth]
    return cropped_data, bbox
