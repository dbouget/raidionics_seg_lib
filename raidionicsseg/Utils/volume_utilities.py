from copy import deepcopy

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

from .configuration_parser import ConfigResources
from .configuration_parser import ImagingModalityType


def input_file_category_disambiguation(input_filename: str) -> str:
    """
    Identifying whether the volume stored on disk under input_filename contains a raw MRI volume or is an integer-like
    volume with labels.
    The category belongs to [MRI, Annotation].

    Parameters
    ----------
    input_filename: str
        Disk location of the volume to disambiguate.

    Returns
    ----------
    str
        Human-readable category identified for the input.
    """
    category = None
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_filename)
    image = reader.Execute()
    # image_type = image.GetPixelIDTypeAsString()
    array = sitk.GetArrayFromImage(image)

    if len(np.unique(array)) > 25 or np.max(array) > 255 or np.min(array) < -1:
        category = "Volume"
    else:
        category = "Annotation"
    return category


def resize_volume(volume, new_slice_size, slicing_plane, order=1):
    new_volume = None
    if not new_slice_size:
        return volume

    if len(new_slice_size) == 2:
        if slicing_plane == "axial":
            new_val = int(volume.shape[2] * (new_slice_size[1] / volume.shape[1]))
            new_volume = resize(volume, (new_slice_size[0], new_slice_size[1], new_val), order=order)
        elif slicing_plane == "sagittal":
            new_val = new_slice_size[0]
            new_volume = resize(volume, (new_val, new_slice_size[0], new_slice_size[1]), order=order)
        elif slicing_plane == "coronal":
            new_val = new_slice_size[0]
            new_volume = resize(volume, (new_slice_size[0], new_val, new_slice_size[1]), order=order)
    elif len(new_slice_size) == 3:
        new_volume = resize(volume, new_slice_size, order=order)
    return new_volume


def __intensity_normalization_CT(volume: np.ndarray, parameters: ConfigResources) -> np.ndarray:
    result = np.copy(volume)

    if parameters.normalization_method == "zeromean":
        mean_val = np.mean(result)
        var_val = np.std(result)
        tmp = (result - mean_val) / var_val
        result = tmp
    elif parameters.normalization_method == "default":
        min_val = np.min(result)
        max_val = np.max(result)
        if (max_val - min_val) != 0:
            result = (result - min_val) / (max_val - min_val)

    return result


def __intensity_normalization_MRI(volume: np.ndarray, parameters: ConfigResources) -> np.ndarray:
    """

    Parameters
    ----------
    volume
    parameters

    Returns
    -------

    """
    result = deepcopy(volume).astype("float32")

    if parameters.normalization_method == "zeromean":
        mean_val = np.mean(result)
        var_val = np.std(result)
        tmp = (result - mean_val) / var_val
        result = tmp
    elif parameters.normalization_method == "zeromean_nonzero":
        slices = result != 0
        masked_img = result[slices]
        mean_val = np.mean(masked_img)
        var_val = np.std(masked_img, ddof=1)
        result[slices] = (masked_img - mean_val) / var_val
    elif parameters.normalization_method == "default":
        min_val = np.min(result)
        max_val = np.max(result)
        if (max_val - min_val) != 0:
            tmp = (result - min_val) / (max_val - min_val)
            result = tmp
    # else:
    #     result = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    return result.astype("float32")


def intensity_normalization(volume: np.ndarray, parameters: ConfigResources) -> np.ndarray:
    if parameters.imaging_modality == ImagingModalityType.CT:
        return __intensity_normalization_CT(volume, parameters)
    elif parameters.imaging_modality == ImagingModalityType.MRI:
        return __intensity_normalization_MRI(volume, parameters)


def intensity_clipping(volume: np.ndarray, parameters: ConfigResources) -> np.ndarray:
    result = deepcopy(volume)
    if parameters.imaging_modality == ImagingModalityType.MRI:
        result[result < 0] = 0  # Soft clipping at 0 for MRI

    if parameters.intensity_clipping_range[1] - parameters.intensity_clipping_range[0] != 100:
        limits = np.percentile(volume, q=parameters.intensity_clipping_range)
        result = np.clip(volume, limits[0], limits[1])
    elif (
        parameters.intensity_clipping_values is not None
        and len(parameters.intensity_clipping_values) != 0
        and parameters.intensity_clipping_values[1] > parameters.intensity_clipping_values[0]
    ):
        result[volume < parameters.intensity_clipping_values[0]] = parameters.intensity_clipping_values[0]
        result[volume > parameters.intensity_clipping_values[1]] = parameters.intensity_clipping_values[1]
    return result


def padding_for_inference(data, slab_size, slicing_plane):
    new_data = deepcopy(data)
    if slicing_plane == "axial":
        missing_dimension = (slab_size - (data.shape[3] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, 0), (0, 0), (0, missing_dimension), (0, 0)), mode="edge")
    elif slicing_plane == "sagittal":
        missing_dimension = (slab_size - (data.shape[1] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, missing_dimension), (0, 0), (0, 0), (0, 0)), mode="edge")
    elif slicing_plane == "coronal":
        missing_dimension = (slab_size - (data.shape[2] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, 0), (0, missing_dimension), (0, 0), (0, 0)), mode="edge")

    return new_data, missing_dimension


def padding_for_inference_both_ends(data, slab_size, slicing_plane):
    new_data = data
    padding_val = int(slab_size / 2)
    if slicing_plane == "axial":
        new_data = np.pad(data, ((0, 0), (0, 0), (0, 0), (padding_val, padding_val), (0, 0)), mode="edge")
    elif slicing_plane == "sagittal":
        new_data = np.pad(data, ((0, 0), (padding_val, padding_val), (0, 0), (0, 0), (0, 0)), mode="edge")
    elif slicing_plane == "coronal":
        new_data = np.pad(data, ((0, 0), (0, 0), (padding_val, padding_val), (0, 0), (0, 0)), mode="edge")

    return new_data


def padding_for_inference_both_ends_patchwise(data, patch_size):
    """
    Input data is supposed to be of shape 5, with (B, X, Y, Z, C) data format.
    """
    new_data = deepcopy(data)

    extra_dims = [0] * 6
    if data.shape[1] <= patch_size[0]:
        padding_val = int(np.ceil(abs(data.shape[1] - patch_size[0]) / 2))
        new_data = np.pad(new_data, ((0, 0), (padding_val, padding_val), (0, 0), (0, 0), (0, 0)), mode="edge")
        extra_dims[0] = padding_val
        extra_dims[1] = padding_val
    if data.shape[2] <= patch_size[1]:
        padding_val = int(np.ceil(abs(data.shape[2] - patch_size[1]) / 2))
        new_data = np.pad(new_data, ((0, 0), (0, 0), (padding_val, padding_val), (0, 0), (0, 0)), mode="edge")
        extra_dims[2] = padding_val
        extra_dims[3] = padding_val
    if data.shape[3] <= patch_size[2]:
        padding_val = int(np.ceil(abs(data.shape[3] - patch_size[2]) / 2))
        new_data = np.pad(new_data, ((0, 0), (0, 0), (0, 0), (padding_val, padding_val), (0, 0)), mode="edge")
        extra_dims[4] = padding_val
        extra_dims[5] = padding_val

    return new_data, extra_dims


def volume_masking(volume, mask, output_filename):
    """
    Masks out everything outside.
    :param volume:
    :param mask:
    :param output_filename:
    :return:
    """
    pass


def volume_cropping(volume, mask, output_filename):
    """
    Crops the initial volume with the tighest bounding box around the mask.
    :param volume:
    :param mask:
    :param output_filename:
    :return:
    """
    pass


def final_activation(x, act_type):
    if act_type == "sigmoid":
        return sigmoid(x)
    else:
        return softmax(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
