import nibabel as nib

from ..Utils.configuration_parser import ConfigResources, ImagingModalityType
from .brain_clipping import crop_MR_background
from .mediastinum_clipping import crop_mediastinum_volume


def extract_foreground(input: nib.Nifti1Image, pre_processing_parameters: ConfigResources, crop_bbox = None):
    res = input
    if pre_processing_parameters.imaging_modality == ImagingModalityType.CT:
        if (
            pre_processing_parameters.crop_background is not None
            and pre_processing_parameters.crop_background != "false"
        ):
            res, crop_bbox = crop_mediastinum_volume(volume=input,
                                                      parameters=pre_processing_parameters, crop_bbox=crop_bbox)
    else:
        if (
            pre_processing_parameters.crop_background is not None
            and not pre_processing_parameters.predictions_use_preprocessed_data
        ):
            res, crop_bbox = crop_MR_background(
                volume=input, parameters=pre_processing_parameters, crop_bbox=crop_bbox)

    return res, crop_bbox