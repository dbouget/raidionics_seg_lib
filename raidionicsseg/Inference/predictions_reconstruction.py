import logging
import traceback
from copy import deepcopy
from typing import List

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import zoom

from ..Utils.configuration_parser import ConfigResources
from ..Utils.resample_or_resize import get_resampler, get_resizer


def reconstruct_post_predictions(
    predictions: np.ndarray,
    parameters: ConfigResources,
    crop_bbox: List[int],
    nib_volume: nib.Nifti1Image,
    fg_volume: nib.Nifti1Image,
    resampled_volume: nib.Nifti1Image,
) -> np.ndarray:
    """
        Reconstructing the inference predictions back into the original patient space.

        Parameters
        ----------
        predictions : np.ndarray
            Results from the inference process.
        parameters : :obj:`ConfigResources`
            Loaded configuration specifying runtime parameters.
        crop_bbox : List[int]
            Indices of a bounding region within the preprocessed volume for additional cropping
            (e.g. coordinates around the brain or lungs).
            The bounding region is expressed as: [minx, miny, minz, maxx, maxy, maxz].
        nib_volume : nib.Nifti1Image
            Original MRI volume, in the patient space, as Nifti format.
        resampled_volume : nib.Nifti1Image
            Processed MRI volume, resampled to output, as Nifti format.

        Returns
        -------
        np.ndarray
    .       Predictions expressed in the original patient space.
    """
    logging.debug("Reconstructing predictions.")
    resampler_type = parameters.resampler
    nb_classes = parameters.training_nb_classes
    final_activation_type = parameters.training_activation_layer_type
    reconstruction_order = parameters.predictions_reconstruction_order
    reconstruction_method = parameters.predictions_reconstruction_method
    probability_thresholds = parameters.training_optimal_thresholds
    swap_input = parameters.swap_training_input

    if reconstruction_order == "resample_first":
        resampled_predictions = __resample_predictions(
            predictions=predictions,
            nb_classes=nb_classes,
            crop_bbox=crop_bbox,
            nib_volume=nib_volume,
            fg_volume=fg_volume,
            resampled_volume=resampled_volume,
            reconstruction_method=reconstruction_method,
            reconstruction_order=reconstruction_order,
            swap_input=swap_input,
            resampler_type=resampler_type,
        )

        final_predictions = __cut_predictions(
            predictions=resampled_predictions,
            reconstruction_method=reconstruction_method,
            probability_threshold=probability_thresholds,
            final_activation_type=final_activation_type,
        )
    else:
        thresh_predictions = __cut_predictions(
            predictions=predictions,
            reconstruction_method=reconstruction_method,
            probability_threshold=probability_thresholds,
            final_activation_type=final_activation_type,
        )
        final_predictions = __resample_predictions(
            predictions=thresh_predictions,
            nb_classes=nb_classes,
            crop_bbox=crop_bbox,
            nib_volume=nib_volume,
            fg_volume=fg_volume,
            resampled_volume=resampled_volume,
            reconstruction_method=reconstruction_method,
            reconstruction_order=reconstruction_order,
            swap_input=swap_input,
            resampler_type=resampler_type,
        )

    return final_predictions


def __cut_predictions(predictions, probability_threshold, reconstruction_method, final_activation_type="softmax"):
    try:
        logging.debug("Clipping predictions with {}.".format(reconstruction_method))
        if reconstruction_method == "probabilities":
            return predictions
        elif reconstruction_method == "thresholding":
            final_predictions = np.zeros(predictions.shape).astype("uint8")
            if len(probability_threshold) != predictions.shape[-1]:
                probability_threshold = np.full(shape=(predictions.shape[-1]), fill_value=probability_threshold[0])

            for c in range(0, predictions.shape[-1]):
                channel = deepcopy(predictions[:, :, :, c])
                channel[channel < probability_threshold[c]] = 0
                channel[channel >= probability_threshold[c]] = 1
                final_predictions[:, :, :, c] = channel.astype("uint8")
        elif reconstruction_method == "argmax":
            if final_activation_type == "sigmoid":
                new_predictions = np.zeros(predictions.shape[:-1] + (predictions.shape[-1] + 1,))
                new_predictions[..., 0][np.max(predictions, axis=-1) <= 0.5] = 1
                new_predictions[..., 1:] = predictions
                final_predictions = np.argmax(new_predictions, axis=-1).astype("uint8")
            else:
                final_predictions = np.argmax(predictions, axis=-1).astype("uint8")
        else:
            raise ValueError("Unknown reconstruction_method with {}!".format(reconstruction_method))
    except Exception as e:
        logging.error(f"Following error collected during predictions clipping: \n {e}\n{traceback.format_exc()}")
        raise ValueError("Predictions clipping process could not fully proceed.")

    return final_predictions


def __resample_predictions(
    predictions,
    nb_classes,
    crop_bbox,
    nib_volume,
    fg_volume,
    resampled_volume,
    reconstruction_method,
    reconstruction_order,
    swap_input,
    resampler_type,
):
    try:
        logging.debug("Resampling predictions with {}.".format(reconstruction_method))
        labels_type = predictions.dtype
        order = 0 if labels_type == np.uint8 else 1
        data = deepcopy(predictions).astype(labels_type)

        if swap_input:
            if len(data.shape) == 4:
                data = np.transpose(data, axes=(1, 0, 2, 3))  # undo transpose
            else:
                data = np.transpose(data, axes=(1, 0, 2))  # undo transpose

        if resampled_volume.shape != predictions.shape[:-1]:
            resizer = get_resizer(type=resampler_type, target_shape=resampled_volume.shape)
            data = resizer.resize(data, resampled_volume.shape)
            # A resize was additionally performed
            # resize_ratio = tuple(np.asarray(resampled_volume.shape) / np.asarray(data.shape[:-1])) + (1.,)
            # data = zoom(data, list(resize_ratio), order=order)


        # Resampling to the size and spacing of the original input volume
        if (
            reconstruction_method == "probabilities"
            or reconstruction_method == "thresholding"
            or (reconstruction_method == "argmax" and reconstruction_order == "resample_first")
        ):
            resampled_predictions = np.zeros(fg_volume.shape + (nb_classes,)).astype(labels_type)
            for c in range(0, nb_classes):
                img = nib.Nifti1Image(data[..., c].astype(labels_type), affine=resampled_volume.affine)
                resampled_channel = resample_from_to(img, fg_volume, order=order)
                resampled_predictions[..., c] = resampled_channel.get_fdata()
        else:
            img = nib.Nifti1Image(data.astype(labels_type), affine=resampled_volume.affine)
            resampled_channel = resample_from_to(img, fg_volume, order=order)
            resampled_predictions = resampled_channel.get_fdata()

        if crop_bbox is not None:
            final_predictions = np.zeros((nib_volume.shape + (nb_classes,)), dtype=labels_type)
            final_predictions[crop_bbox[0]: crop_bbox[3], crop_bbox[1]: crop_bbox[4], crop_bbox[2]: crop_bbox[5]] = resampled_predictions
        else:
            final_predictions = resampled_predictions

        # Range has to be set to [0, 1] again after resampling with interpolation
        if order == 3:
            for c in range(0, nb_classes):
                min_val = np.min(final_predictions[..., c])
                max_val = np.max(final_predictions[..., c])

                if (max_val - min_val) != 0:
                    final_predictions[..., c] = (final_predictions[..., c] - min_val) / (max_val - min_val)
    except Exception as e:
        logging.error(f"Following error collected during predictions resampling: \n {e}\n{traceback.format_exc()}")
        raise ValueError("Predictions resampling process could not fully proceed.")

    return final_predictions
