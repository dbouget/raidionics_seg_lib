import logging
import os
from copy import deepcopy
from pathlib import PurePath
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from nibabel.processing import resample_to_output
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label
from skimage.measure import regionprops

from ..Utils.configuration_parser import ConfigResources
from ..Utils.io import load_nifti_volume


def crop_MR_background(
    volume: np.ndarray,
    new_spacing: Tuple[float],
    parameters: ConfigResources,
    crop_bbox: List[int] = None,
) -> Tuple[np.ndarray, List[int]]:
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
        Destination folder where the results will be stored.
    parameters :  :obj:`ConfigResources`
        Loaded configuration specifying runtime parameters.
    Returns
    -------
    np.ndarray
        New volume after background cropping procedure.
    List[int]
        Indices of a bounding region within the volume for additional cropping (e.g. coordinates around the head,
        or tightly around the brain only).
        The bounding region is expressed as: [minx, miny, minz, maxx, maxy, maxz].
    """
    if parameters.crop_background == "minimum":
        return crop_background_minimum(volume=volume, crop_bbox=crop_bbox)
    elif parameters.crop_background == "brain_clip" or parameters.crop_background == "brain_mask":
        return skull_stripping(volume=volume, new_spacing=new_spacing, parameters=parameters)


def crop_background_minimum(
    volume: np.ndarray, crop_bbox: Union[None, List[int]] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Performs background cropping inside an MRI volume in 'minimum' mode whereby the black space around
    the head is removed.

    Parameters
    ----------
    volume : np.ndarray
        Patient MRI volume to crop.
    crop_bbox:
    Returns
    -------
    np.ndarray
        New volume after background cropping procedure.
    List[int]
        Indices of a bounding region within the volume parametrizing the cropping.
        The bounding region is expressed as: [minx, miny, minz, maxx, maxy, maxz].
    """
    original_volume = np.copy(volume)
    if crop_bbox is None:
        limit = int((np.max(volume) - np.min(volume)) * 0.2)
        volume[original_volume >= limit] = 1
        volume[original_volume < limit] = 0
        volume = volume.astype(np.uint8)
        volume = binary_fill_holes(volume).astype(np.uint8)
        regions = regionprops(volume)
        min_row, min_col, min_depth, max_row, max_col, max_depth = regions[0].bbox
        cropped_volume = original_volume[min_row:max_row, min_col:max_col, min_depth:max_depth]
        crop_bbox = [min_row, min_col, min_depth, max_row, max_col, max_depth]
    else:
        min_row, min_col, min_depth, max_row, max_col, max_depth = (
            crop_bbox[0],
            crop_bbox[1],
            crop_bbox[2],
            crop_bbox[3],
            crop_bbox[4],
            crop_bbox[5],
        )
        cropped_volume = original_volume[min_row:max_row, min_col:max_col, min_depth:max_depth]

    logging.debug(
        "MRI background cropping with: [{}, {}, {}, {}, {}, {}].\n".format(
            min_row, min_col, min_depth, max_row, max_col, max_depth
        )
    )
    return cropped_volume, crop_bbox


def skull_stripping(
    volume: np.ndarray, new_spacing: Tuple[float], parameters: ConfigResources
) -> Tuple[np.ndarray, List[int]]:
    """
    Performs skull stripping over the provided input volume using the brain mask manually provided.

    Parameters
    ----------
    volume : np.ndarray
        Patient MRI volume to skull strip.
    new_spacing : Tuple[float]
        .
    parameters :  :obj:`ConfigResources`
        Loaded configuration specifying runtime parameters.
    Returns
    -------
    np.ndarray
        New volume after background cropping procedure.
    List[int]
        Indices of a bounding region within the volume parametrizing the cropping.
        The bounding region is expressed as: [minx, miny, minz, maxx, maxy, maxz].
    """
    if not os.path.exists(parameters.runtime_brain_mask_filepath):
        raise ValueError("A brain segmentation mask must be provided inside ['Neuro']['brain_segmentation_filename']")
    else:
        brain_mask_filename = parameters.runtime_brain_mask_filepath

    brain_mask_ni = load_nifti_volume(brain_mask_filename)
    resampled_volume = resample_to_output(brain_mask_ni, new_spacing, order=0)
    brain_mask = resampled_volume.get_fdata().astype("uint8")

    # In case of noisy segmentation (which should not happen), only the biggest component is kept.
    labels, nb_components = label(brain_mask)
    brain_objects_properties = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)
    brain_object = brain_objects_properties[0]
    brain_component = np.zeros(brain_mask.shape).astype("uint8")
    brain_component[
        brain_object.bbox[0] : brain_object.bbox[3],
        brain_object.bbox[1] : brain_object.bbox[4],
        brain_object.bbox[2] : brain_object.bbox[5],
    ] = 1

    final_brain_mask = brain_mask & brain_component

    return advanced_crop_exclude_background(volume, crop_mode=parameters.crop_background, brain_mask=final_brain_mask)


def advanced_crop_exclude_background(
    volume: np.ndarray, crop_mode: str, brain_mask: np.ndarray
) -> Tuple[np.ndarray, List[int]]:
    """
    Performs skull stripping either in mode 'brain_clip' or mode 'brain_mask'.

    Parameters
    ----------
    volume : np.ndarray
        Patient MRI volume to skull strip.
    crop_mode : str
        Parameters defining the skull-stripping, in 'brain_mask' mode all voxels not belonging to the
        brain are set to rgb(0, 0, 0), and in 'brain_clip' mode only the smallest region around the brain is kept.
    brain_mask: np.ndarray
        Binary image where voxels belonging to the brain category are set to 1, and the rest to 0.
    Returns
    -------
    np.ndarray
        New volume after background cropping procedure.
    List[int]
        Indices of a bounding region within the volume parametrizing the cropping.
        The bounding region is expressed as: [minx, miny, minz, maxx, maxy, maxz].
    """
    original_data = np.copy(volume)
    regions = regionprops(brain_mask)
    min_row, min_col, min_depth, max_row, max_col, max_depth = regions[0].bbox

    if crop_mode == "brain_mask":
        original_data[brain_mask == 0] = 0

    cropped_data = original_data[min_row:max_row, min_col:max_col, min_depth:max_depth]
    bbox = [min_row, min_col, min_depth, max_row, max_col, max_depth]
    logging.debug(
        "MRI skull stripping with: [{}, {}, {}, {}, {}, {}].".format(
            min_row, min_col, min_depth, max_row, max_col, max_depth
        )
    )
    return cropped_data, bbox
