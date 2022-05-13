import numpy as np
from copy import deepcopy
from skimage.transform import resize
from skimage.measure import regionprops
from .configuration_parser import *


def resize_volume(volume, new_slice_size, slicing_plane, order=1):
    new_volume = None
    if len(new_slice_size) == 2:
        if slicing_plane == 'axial':
            new_val = int(volume.shape[2] * (new_slice_size[1] / volume.shape[1]))
            new_volume = resize(volume, (new_slice_size[0], new_slice_size[1], new_val), order=order)
        elif slicing_plane == 'sagittal':
            new_val = new_slice_size[0]
            new_volume = resize(volume, (new_val, new_slice_size[0], new_slice_size[1]), order=order)
        elif slicing_plane == 'coronal':
            new_val = new_slice_size[0]
            new_volume = resize(volume, (new_slice_size[0], new_val, new_slice_size[1]), order=order)
    elif len(new_slice_size) == 3:
        new_volume = resize(volume, new_slice_size, order=order)
    return new_volume


def __intensity_normalization_CT(volume, parameters):
    result = np.copy(volume)

    result[volume < parameters.intensity_clipping_values[0]] = parameters.intensity_clipping_values[0]
    result[volume > parameters.intensity_clipping_values[1]] = parameters.intensity_clipping_values[1]

    min_val = np.min(result)
    max_val = np.max(result)
    if (max_val - min_val) != 0:
        result = (result - min_val) / (max_val - min_val)

    return result


def __intensity_normalization_MRI(volume, parameters):
    result = deepcopy(volume).astype('float32')
    result[result < 0] = 0  # Soft clipping at 0 for MRI
    if parameters.intensity_clipping_range[1] - parameters.intensity_clipping_range[0] != 100:
        limits = np.percentile(volume, q=parameters.intensity_clipping_range)
        result[volume < limits[0]] = limits[0]
        result[volume > limits[1]] = limits[1]

    if parameters.normalization_method == 'zeromean':
        mean_val = np.mean(result)
        var_val = np.std(result)
        tmp = (result - mean_val) / var_val
        result = tmp
    elif parameters.normalization_method == 'default':
        min_val = np.min(result)
        max_val = np.max(result)
        if (max_val - min_val) != 0:
            tmp = (result - min_val) / (max_val - min_val)
            result = tmp
    # else:
    #     result = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    return result


def intensity_normalization(volume, parameters):
    if parameters.imaging_modality == ImagingModalityType.CT:
        return __intensity_normalization_CT(volume, parameters)
    elif parameters.imaging_modality == ImagingModalityType.MRI:
        return __intensity_normalization_MRI(volume, parameters)


def padding_for_inference(data, slab_size, slicing_plane):
    new_data = data
    if slicing_plane == 'axial':
        missing_dimension = (slab_size - (data.shape[2] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, 0), (0, missing_dimension), (0, 0)), mode='edge')
    elif slicing_plane == 'sagittal':
        missing_dimension = (slab_size - (data.shape[0] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, missing_dimension), (0, 0), (0, 0), (0, 0)), mode='edge')
    elif slicing_plane == 'coronal':
        missing_dimension = (slab_size - (data.shape[1] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, missing_dimension), (0, 0), (0, 0)), mode='edge')

    return new_data, missing_dimension


def padding_for_inference_both_ends(data, slab_size, slicing_plane):
    new_data = data
    padding_val = int(slab_size / 2)
    if slicing_plane == 'axial':
        new_data = np.pad(data, ((0, 0), (0, 0), (padding_val, padding_val), (0, 0)), mode='edge')
    elif slicing_plane == 'sagittal':
        new_data = np.pad(data, ((padding_val, padding_val), (0, 0), (0, 0), (0, 0)), mode='edge')
    elif slicing_plane == 'coronal':
        new_data = np.pad(data, ((0, 0), (padding_val, padding_val), (0, 0), (0, 0)), mode='edge')

    return new_data


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
