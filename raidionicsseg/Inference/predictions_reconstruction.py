import logging
import time
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
    system_acceleration = parameters.system_acceleration
    nb_classes = parameters.training_nb_classes
    final_activation_type = parameters.training_activation_layer_type
    reconstruction_order = parameters.predictions_reconstruction_order
    reconstruction_method = parameters.predictions_reconstruction_method
    probability_thresholds = parameters.training_optimal_thresholds
    swap_input = parameters.swap_training_input
    device_id = parameters.gpu_id

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
            system_acceleration=system_acceleration,
        )

        final_predictions = __cut_predictions(
            predictions=resampled_predictions,
            reconstruction_method=reconstruction_method,
            probability_threshold=probability_thresholds,
            final_activation_type=final_activation_type,
            system_acceleration=system_acceleration,
            device_id=device_id
        )
    else:
        thresh_predictions = __cut_predictions(
            predictions=predictions,
            reconstruction_method=reconstruction_method,
            probability_threshold=probability_thresholds,
            final_activation_type=final_activation_type,
            system_acceleration=system_acceleration,
            device_id=device_id
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
            system_acceleration=system_acceleration,
        )

    return final_predictions


def __cut_predictions(predictions, probability_threshold, reconstruction_method, final_activation_type="softmax",
                      system_acceleration="cpu", device_id="-1"):
    try:
        logging.debug(f"Clipping predictions with {reconstruction_method}.")
        if system_acceleration == "torch":
            return __cut_predictions_torch(predictions=predictions, probability_threshold=probability_threshold,
                                           reconstruction_method=reconstruction_method,
                                           final_activation_type=final_activation_type, device_id=device_id)

        if reconstruction_method == "probabilities":
            return predictions
        elif reconstruction_method == "thresholding":
            if len(probability_threshold) != predictions.shape[-1]:
                probability_threshold = np.full(shape=(predictions.shape[-1]), fill_value=probability_threshold[0])
            final_predictions = (predictions >= probability_threshold).astype("uint8")

        elif reconstruction_method == "argmax":
            if final_activation_type == "sigmoid":
                # new array (H,W,D,C+1)
                # Background channel is 1 where max(pred) <= 0.5
                bg = (np.max(predictions, axis=-1) <= 0.5).astype(predictions.dtype)
                bg = bg[..., None]  # shape expand

                # Stack background + predictions
                new_predictions = np.concatenate([bg, predictions], axis=-1)

                final_predictions = np.argmax(new_predictions, axis=-1).astype("uint8")
                # new_predictions = np.zeros(predictions.shape[:-1] + (predictions.shape[-1] + 1,))
                # new_predictions[..., 0][np.max(predictions, axis=-1) <= 0.5] = 1
                # new_predictions[..., 1:] = predictions
                # final_predictions = np.argmax(new_predictions, axis=-1).astype("uint8")
            else:
                final_predictions = np.argmax(predictions, axis=-1).astype("uint8")
        else:
            raise ValueError(f"Unknown reconstruction_method with {reconstruction_method}!")
    except Exception as e:
        logging.error(f"Following error collected during predictions clipping: \n {e}\n{traceback.format_exc()}")
        raise ValueError("Predictions clipping process could not fully proceed.")

    return final_predictions


def __cut_predictions_torch(predictions,
                          probability_threshold,
                          reconstruction_method,
                          final_activation_type="softmax",
                          device_id="0"):
    """
    predictions: NumPy array or torch.Tensor of shape (H,W,D,C)
    returns: NumPy uint8 array
    """
    import torch

    device = f"cuda:{device_id}"
    if isinstance(predictions, np.ndarray):
        preds = torch.from_numpy(predictions).to(device)
    else:
        preds = predictions.to(device)

    C = preds.shape[-1]
    if reconstruction_method == "thresholding":
        if len(probability_threshold) != C:
            th = torch.tensor([probability_threshold[0]] * C, device=device).view(1,1,1,C)
        else:
            th = torch.tensor(probability_threshold, device=device).view(1,1,1,C)

    # -------------------------------------------------------------------------
    # Probabilities
    # -------------------------------------------------------------------------
    if reconstruction_method == "probabilities":
        return preds.detach().cpu().numpy()

    # -------------------------------------------------------------------------
    # Thresholding (vectorized)
    # -------------------------------------------------------------------------
    elif reconstruction_method == "thresholding":
        final_predictions = (preds >= th).to(torch.uint8)
        return final_predictions.cpu().numpy()

    # -------------------------------------------------------------------------
    # Argmax
    # -------------------------------------------------------------------------
    elif reconstruction_method == "argmax":

        if final_activation_type == "sigmoid":
            # Create extra class channel 0:
            # If max prob ≤ 0.5 → background
            max_val = torch.max(preds, dim=-1, keepdim=True).values
            bg = (max_val <= 0.5).to(preds.dtype)

            # new shape: (H,W,D,C+1)
            new_preds = torch.cat([bg, preds], dim=-1)

            final_predictions = torch.argmax(new_preds, dim=-1).to(torch.uint8)
            return final_predictions.cpu().numpy()

        else:
            final_predictions = torch.argmax(preds, dim=-1).to(torch.uint8)
            return final_predictions.cpu().numpy()

    else:
        raise ValueError(f"Unknown reconstruction_method '{reconstruction_method}'")


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
    system_acceleration,
):
    try:
        logging.debug("Resampling predictions with {}.".format(reconstruction_method))
        start = time.time()
        labels_type = predictions.dtype
        order = 0 if labels_type == np.uint8 else 1
        data = deepcopy(predictions).astype(labels_type)

        if swap_input:
            if len(data.shape) == 4:
                data = np.transpose(data, axes=(1, 0, 2, 3))  # undo transpose
            else:
                data = np.transpose(data, axes=(1, 0, 2))  # undo transpose

        if resampled_volume.shape != predictions.shape[:-1]:
            resizer = get_resizer(type=system_acceleration, target_shape=resampled_volume.shape, order=order)
            data = resizer.resize(data, resampled_volume.shape)

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

    logging.debug(f"Predictions resampling took: {time.time() - start}")
    return final_predictions
