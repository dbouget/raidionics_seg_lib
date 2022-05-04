import logging
from typing import Tuple, Any
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from dipy.align.reslice import reslice  # @TODO. Is dipy really needed?
from raidionicsseg.Utils.volume_utilities import intensity_normalization, resize_volume
from raidionicsseg.Utils.io import load_nifti_volume
from raidionicsseg.PreProcessing.mediastinum_clipping import mediastinum_clipping, mediastinum_clipping_DL
from raidionicsseg.PreProcessing.brain_clipping import crop_MR_background
from raidionicsseg.Utils.configuration_parser import ImagingModalityType
from raidionicsseg.Utils.configuration_parser import ConfigResources


def run_pre_processing(filename: str, pre_processing_parameters: ConfigResources,
                       storage_path: str) -> Tuple[nib.Nifti1Image, np.ndarray, np.ndarray, Any]:
    """

    Parameters
    ----------
    filename : str
        The first parameter.
    pre_processing_parameters : :obj:`ConfigResources`
        The second parameter.
    storage_path: str
        The third parameter.

    Returns
    -------
    nib.Nifti1Image
        First return
    np.ndarray
        Second return
    np.ndarray
        Third return
    Any
        Fourth return
    """
    logging.info("Extracting input data.\n")
    nib_volume = load_nifti_volume(filename)

    logging.info("Resampling.\n")
    # Normalize spacing
    new_spacing = pre_processing_parameters.output_spacing
    if pre_processing_parameters.output_spacing == None:
        tmp = np.min(nib_volume.header.get_zooms())
        new_spacing = [tmp, tmp, tmp]

    library = pre_processing_parameters.preprocessing_library
    if library == 'nibabel':
        resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
        data = resampled_volume.get_data().astype('float32')
    elif library == 'dipy':
        data, aff = reslice(nib_volume.get_data(), nib_volume.affine, nib_volume.header.get_zooms(), new_zooms=new_spacing)
        resampled_volume = nib.Nifti1Image(data, affine=aff)

    logging.info("Clipping and intensity normalization.\n")
    crop_bbox = None
    if pre_processing_parameters.imaging_modality == ImagingModalityType.CT:
        # Exclude background
        if pre_processing_parameters.crop_background is not None and pre_processing_parameters.crop_background != 'false':
                #data, crop_bbox = mediastinum_clipping(volume=data, parameters=pre_processing_parameters)
                data, crop_bbox = mediastinum_clipping_DL(filename, data, new_spacing, storage_path, pre_processing_parameters)
        # Normalize values
        data = intensity_normalization(volume=data, parameters=pre_processing_parameters)
        # set intensity range
        mins, maxs = pre_processing_parameters.intensity_target_range
        data *= maxs  # @TODO: Modify it such that it handles scaling to i.e. [-1, -1]
    else:
        # Normalize values
        data = intensity_normalization(volume=data, parameters=pre_processing_parameters)
        # Exclude background
        if pre_processing_parameters.crop_background is not None:
            data, crop_bbox = crop_MR_background(filename, data, new_spacing, storage_path, pre_processing_parameters)

    logging.info("Volume resizing.\n")
    data = resize_volume(data, pre_processing_parameters.new_axial_size, pre_processing_parameters.slicing_plane,
                         order=1)

    if pre_processing_parameters.swap_training_input:
        data = np.transpose(data, axes=(1, 0, 2))

    return nib_volume, resampled_volume, data, crop_bbox
