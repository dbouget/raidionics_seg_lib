import configparser
import os
from copy import deepcopy
from typing import List
from typing import Tuple

import numpy as np
import scipy.ndimage.measurements as smeas
import scipy.ndimage.morphology as smo
from nibabel.processing import resample_to_output
from skimage.measure import regionprops

from ..Utils.configuration_parser import ConfigResources
from ..Utils.io import load_nifti_volume


def crop_mediastinum_volume(
    filepath: str, volume: np.ndarray, new_spacing: Tuple[float], storage_path: str, parameters: ConfigResources
) -> Tuple[np.ndarray, List[int]]:
    """
    Performs different background cropping inside a mediastinal CT volume, as defined by the 'crop_background' stored
    in a model preprocessing configuration file.
    In 'minimum' mode, the black space around the head is removed, in 'lungs_mask' mode all voxels not belonging to the
    brain are set to rgb(0, 0, 0), and in 'lungs_clip' mode only the smallest bounding region around the brain is kept.

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
        return mediastinum_clipping(volume, parameters)
    elif parameters.crop_background == "brain_clip" or parameters.crop_background == "brain_mask":
        return mediastinum_clipping_DL(filepath, volume, new_spacing, storage_path, parameters)


def mediastinum_clipping(volume, parameters):
    intensity_threshold = -250
    airmetal_mask = deepcopy(volume)
    airmetal_mask[airmetal_mask > intensity_threshold] = 0
    airmetal_mask[airmetal_mask <= intensity_threshold] = 1

    airmetal_mask = smo.binary_closing(airmetal_mask, iterations=5)

    labels, nb_components = smeas.label(airmetal_mask)
    airmetal_pieces = smeas.find_objects(labels, min(nb_components, 1000))

    nums = []
    for p in enumerate(airmetal_pieces):
        bb = p[1]
        z = bb[2].stop - bb[2].start
        y = bb[1].stop - bb[1].start
        x = bb[0].stop - bb[0].start
        nums.append(x * y * z)

    # Should check if the first two or three elements are "as big". If two the following code is correct, if three
    # something should be changed so that the lungs is the third (normally?) and background the first two.
    ind_bg = nums.index(np.max(nums)) + 1
    nums.remove(np.max(nums))
    ind_lungs = (
        nums.index(np.max(nums)) + 1 + 1
    )  # +1 because find_objects labels start at 1, and +1 because we remove one value above
    # ind_lungs = nums.index(sorted(nums, reverse=True)[0])+1
    # for l in range(nb_components):
    #   nums.append(np.count_nonzero(np.where(labels == l)))

    background_mask = np.copy(labels)
    background_mask[background_mask != ind_bg] = 0

    lungstrachea_mask = np.copy(labels)
    lungstrachea_mask[lungstrachea_mask != ind_lungs] = 0
    lungstrachea_mask[lungstrachea_mask == ind_lungs] = 1

    lungs_boundingbox = airmetal_pieces[ind_lungs - 1]  # Because indexing starts at 0, so have to decrease by one
    crop_bbox = [
        lungs_boundingbox[0].start,
        lungs_boundingbox[1].start,
        lungs_boundingbox[2].start,
        lungs_boundingbox[0].stop,
        lungs_boundingbox[1].stop,
        lungs_boundingbox[2].stop,
    ]

    cropped_volume = volume[crop_bbox[0] : crop_bbox[3], crop_bbox[1] : crop_bbox[4], crop_bbox[2] : crop_bbox[5]]

    print("Cropped mediastinum values: {}".format(lungs_boundingbox))
    return cropped_volume, crop_bbox


def mediastinum_clipping_DL(filepath, volume, new_spacing, storage_path, parameters):
    if not parameters.runtime_lungs_mask_filepath and not os.path.exists(parameters.runtime_lungs_mask_filepath):
        lung_config_filename = os.path.join(os.path.dirname(parameters.config_filename), "lungs_main_config.ini")
        new_parameters = configparser.ConfigParser()
        new_parameters.read(parameters.config_filename)
        new_parameters.set("System", "model_folder", os.path.join(os.path.dirname(parameters.model_folder), "CT_Lungs"))
        new_parameters.set("Runtime", "reconstruction_method", "thresholding")
        new_parameters.set("Runtime", "reconstruction_order", "resample_first")
        with open(lung_config_filename, "w") as cf:
            new_parameters.write(cf)
        old_parameters = deepcopy(parameters)
        from raidionicsseg.fit import run_model

        run_model(lung_config_filename)
        lungs_mask_filename = os.path.join(storage_path, "labels_Lungs.nii.gz")
        os.remove(lung_config_filename)
        parameters = old_parameters
    else:
        lungs_mask_filename = parameters.runtime_lungs_mask_filepath

    lungs_mask_ni = load_nifti_volume(lungs_mask_filename)
    resampled_volume = resample_to_output(lungs_mask_ni, new_spacing, order=0)
    lungs_mask = resampled_volume.get_fdata().astype("uint8")
    # In case the lungs mask has a different label for each lung
    lungs_mask[lungs_mask != 0] = 1

    lung_region = regionprops(lungs_mask)
    min_row, min_col, min_depth, max_row, max_col, max_depth = lung_region[0].bbox
    if parameters.crop_background == "invert":
        max_depth = min_depth
        min_depth = 0
    print("cropping params", min_row, min_col, min_depth, max_row, max_col, max_depth)

    cropped_volume = volume[min_row:max_row, min_col:max_col, min_depth:max_depth]
    bbox = [min_row, min_col, min_depth, max_row, max_col, max_depth]

    return cropped_volume, bbox
