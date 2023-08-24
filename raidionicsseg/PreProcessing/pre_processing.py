import logging
import os
from typing import Tuple, Any, List
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from ..Utils.volume_utilities import intensity_normalization, resize_volume, input_file_category_disambiguation
from ..Utils.io import load_nifti_volume
from .mediastinum_clipping import mediastinum_clipping, mediastinum_clipping_DL
from .brain_clipping import crop_MR_background
from ..Utils.configuration_parser import ConfigResources, ImagingModalityType


def prepare_pre_processing(folder: str, pre_processing_parameters: ConfigResources,
                           storage_path: str) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray, List[int]]:
    # input_files = []
    # for _, _, files in os.walk(folder):
    #     for f in files:
    #         input_files.append(f)
    #     break

    input_file = os.path.join(folder, 'input0.nii.gz')
    nib_volume, resampled_volume, data, crop_bbox = run_pre_processing(input_file, pre_processing_parameters,
                                                                       storage_path)
    final_data = np.zeros((1,) + data.shape + (pre_processing_parameters.preprocessing_number_inputs,)).astype('float32')
    final_data[..., 0] = data
    for i in range(1, pre_processing_parameters.preprocessing_number_inputs):
        input_file = os.path.join(folder, 'input' + str(i) + '.nii.gz')
        _, _, data, _ = run_pre_processing(input_file, pre_processing_parameters, storage_path)
        final_data[..., i] = data

    return nib_volume, resampled_volume, final_data, crop_bbox


def run_pre_processing(filename: str, pre_processing_parameters: ConfigResources,
                       storage_path: str) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray, List[int]]:
    """

    Parameters
    ----------
    filename : str
        Filepath of the input volume (CT or MRI) to use.
    pre_processing_parameters : :obj:`ConfigResources`
        Loaded configuration specifying runtime parameters.
    storage_path: str
        Folder where the computed results should be stored.

    Returns
    -------
    nib.Nifti1Image
        Original Nifti object from loading the content of filename.
    nib.Nifti1Image
        Nifti object after conversion to a normalized space (resample_to_output).
    np.ndarray
        Fully preprocessed volume ready for inference.
    List[int]
        Indices of a bounding region within the preprocessed volume for additional cropping
         (e.g. coordinates around the brain or lungs).
         The bounding region is expressed as: [minx, miny, minz, maxx, maxy, maxz].
    """
    logging.debug("Preprocessing - Extracting input data.")
    nib_volume = load_nifti_volume(filename)
    input_category = input_file_category_disambiguation(filename)
    processing_order = 1
    if input_category == 'Annotation':
        processing_order = 0

    logging.debug("Preprocessing - Resampling.")
    new_spacing = pre_processing_parameters.output_spacing
    if pre_processing_parameters.output_spacing == None:
        tmp = np.min(nib_volume.header.get_zooms())
        new_spacing = [tmp, tmp, tmp]

    resampled_volume = resample_to_output(nib_volume, new_spacing, order=processing_order)
    data = resampled_volume.get_fdata().astype('float32')

    logging.debug("Preprocessing - Background clipping.")
    crop_bbox = None
    if pre_processing_parameters.imaging_modality == ImagingModalityType.CT:
        # Exclude background
        if pre_processing_parameters.crop_background is not None and pre_processing_parameters.crop_background != 'false':
                #data, crop_bbox = mediastinum_clipping(volume=data, parameters=pre_processing_parameters)
                data, crop_bbox = mediastinum_clipping_DL(filename, data, new_spacing, storage_path,
                                                          pre_processing_parameters)
    else:
        if pre_processing_parameters.crop_background is not None and \
                not pre_processing_parameters.predictions_use_preprocessed_data:
            data, crop_bbox = crop_MR_background(filename, data, new_spacing, storage_path, pre_processing_parameters)

    if pre_processing_parameters.new_axial_size:
        logging.debug("Preprocessing - Volume resizing.")
        data = resize_volume(data, pre_processing_parameters.new_axial_size, pre_processing_parameters.slicing_plane,
                             order=processing_order)
    if input_category == 'Volume':
        # Normalize values
        logging.debug("Preprocessing - Intensity normalization.")
        data = intensity_normalization(volume=data, parameters=pre_processing_parameters)

    if pre_processing_parameters.swap_training_input:
        data = np.transpose(data, axes=(1, 0, 2))

    return nib_volume, resampled_volume, data, crop_bbox
