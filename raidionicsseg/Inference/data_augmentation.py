import numpy as np
import random
import logging
from typing import List
from copy import deepcopy
from abc import ABC, abstractmethod
from scipy.ndimage import zoom, rotate
from ..Utils.configuration_parser import ImagingModalityType


class Transform(ABC):
    """
    Abstract class for data augmentation transforms.
    Inheriting transform classes should not use other high-level libraries (e.g. imgaug, torchio) to not make the
    wheel any bigger than necessary and to avoid Raidionics packaging conflict with opencv-python. As a result, the
    process is not speed-efficient right now.
    """
    @abstractmethod
    def transform(self, data: np.ndarray, direction: str) -> np.ndarray:
        """
        Abstract class performing the data augmentation transform.

        Parameters
        ----------
        data: preprocessed input data to augment
        direction: string indicating if the transformation should be performed or reverted, to sample from: [forward, inverse]

        Returns
        -------
        np.ndarray
            Augmented input data stored as an array and compatible with the inference command
        """
        pass

    @abstractmethod
    def randomize(self) -> None:
        """
        Abstract method for performing the internal randomization process for the probability and parameters.
        Returns
        -------

        """
        pass


class Flip(Transform):
    def __init__(self):
        self.prob_flip1 = 0
        self.prob_flip2 = 0
        self.prob_flip3 = 0
        self.randomize()

    def randomize(self):
        self.prob_flip1 = random.uniform(0, 1)
        self.prob_flip2 = random.uniform(0, 1)
        self.prob_flip3 = random.uniform(0, 1)

    def transform(self, data, direction):
        data_aug = deepcopy(data)
        if direction == "forward":
            if self.prob_flip1 > 0.5:
                data_aug = data_aug[:, ::-1, :, :, :]
            if self.prob_flip2 > 0.5:
                data_aug = data_aug[:, :, ::-1, :, :]
            if self.prob_flip3 > 0.5:
                data_aug = data_aug[:, :, :, ::-1, :]
        else:
            if self.prob_flip3 > 0.5:
                data_aug = data_aug[:, :, ::-1, :]
            if self.prob_flip2 > 0.5:
                data_aug = data_aug[:, ::-1, :, :]
            if self.prob_flip1 > 0.5:
                data_aug = data_aug[::-1, :, :, :]

        return data_aug


class Zoom(Transform):
    def __init__(self):
        self.prob_zoom = 0
        self.zoom_ratio = 0
        self.randomize()

    def randomize(self):
        self.prob_zoom = random.uniform(0, 1)
        self.zoom_ratio = random.uniform(0.75, 1.5)

    def transform(self, data, direction):
        data_aug = deepcopy(data)
        if direction == "forward":
            zoom_ratio_axis = [self.zoom_ratio] * (len(data.shape) - 2) + [1.]
            if self.prob_zoom > 0.:
                data_aug = zoom(data_aug[0, ...], zoom_ratio_axis)
                data_aug = np.expand_dims(data_aug, axis=0)
                if self.zoom_ratio > 1.:
                    crop_data_aug = np.zeros(data.shape, dtype=np.float32)
                    crop_limits = [int((data_aug.shape[1] - data.shape[1])/2),
                                   (data_aug.shape[1] - data.shape[1]) - int((data_aug.shape[1] - data.shape[1])/2),
                                   int((data_aug.shape[2] - data.shape[2]) / 2),
                                   (data_aug.shape[2] - data.shape[2]) - int((data_aug.shape[2] - data.shape[2]) / 2),
                                   int((data_aug.shape[3] - data.shape[3]) / 2),
                                   (data_aug.shape[3] - data.shape[3]) - int((data_aug.shape[3] - data.shape[3]) / 2)
                                   ]
                    crop_data_aug[:] = data_aug[:, crop_limits[0]:data_aug.shape[1] - crop_limits[1],
                                       crop_limits[2]:data_aug.shape[2] - crop_limits[3],
                                       crop_limits[4]:data_aug.shape[3] - crop_limits[5], :]
                    data_aug = crop_data_aug
                elif self.zoom_ratio < 1.:
                    pad_data_aug = np.zeros(data.shape, dtype=np.float32)
                    pad_limits = [int((data.shape[1] - data_aug.shape[1]) / 2),
                                   (data.shape[1] - data_aug.shape[1]) - int((data.shape[1] - data_aug.shape[1]) / 2),
                                   int((data.shape[2] - data_aug.shape[2]) / 2),
                                   (data.shape[2] - data_aug.shape[2]) - int((data.shape[2] - data_aug.shape[2]) / 2),
                                   int((data.shape[3] - data_aug.shape[3]) / 2),
                                   (data.shape[3] - data_aug.shape[3]) - int((data.shape[3] - data_aug.shape[3]) / 2)
                                   ]
                    pad_data_aug[:, pad_limits[0]:data.shape[1] - pad_limits[1],
                                       pad_limits[2]:data.shape[2] - pad_limits[3],
                                       pad_limits[4]:data.shape[3] - pad_limits[5], :] = data_aug[:]
                    data_aug = pad_data_aug
        else:
            if self.prob_zoom > 0.:
                zoom_ratio_axis = [1/self.zoom_ratio] * (len(data.shape) - 1) + [1.]
                data_aug = zoom(data_aug, zoom_ratio_axis)
                if self.zoom_ratio < 1.:
                    crop_data_aug = np.zeros(data.shape, dtype=np.float32)
                    crop_limits = [int((data_aug.shape[0] - data.shape[0])/2),
                                   (data_aug.shape[0] - data.shape[0]) - int((data_aug.shape[0] - data.shape[0])/2),
                                   int((data_aug.shape[1] - data.shape[1]) / 2),
                                   (data_aug.shape[1] - data.shape[1]) - int((data_aug.shape[1] - data.shape[1]) / 2),
                                   int((data_aug.shape[2] - data.shape[2]) / 2),
                                   (data_aug.shape[2] - data.shape[2]) - int((data_aug.shape[2] - data.shape[2]) / 2)
                                   ]
                    crop_data_aug[:] = data_aug[crop_limits[0]:data_aug.shape[0] - crop_limits[1],
                                       crop_limits[2]:data_aug.shape[1] - crop_limits[3],
                                       crop_limits[4]:data_aug.shape[2] - crop_limits[5], :]
                    data_aug = crop_data_aug
                elif self.zoom_ratio > 1.:
                    pad_data_aug = np.zeros(data.shape, dtype=np.float32)
                    pad_limits = [int((data.shape[0] - data_aug.shape[0]) / 2),
                                   (data.shape[0] - data_aug.shape[0]) - int((data.shape[0] - data_aug.shape[0]) / 2),
                                   int((data.shape[1] - data_aug.shape[1]) / 2),
                                   (data.shape[1] - data_aug.shape[1]) - int((data.shape[1] - data_aug.shape[1]) / 2),
                                   int((data.shape[2] - data_aug.shape[2]) / 2),
                                   (data.shape[2] - data_aug.shape[2]) - int((data.shape[2] - data_aug.shape[2]) / 2)
                                   ]
                    pad_data_aug[pad_limits[0]:data.shape[0] - pad_limits[1],
                                       pad_limits[2]:data.shape[1] - pad_limits[3],
                                       pad_limits[4]:data.shape[2] - pad_limits[5], :] = data_aug[:]
                    data_aug = pad_data_aug
        return data_aug


class Rotate(Transform):
    def __init__(self):
        self.prob_rotate1 = 0
        self.prob_rotate2 = 0
        self.prob_rotate3 = 0
        self.rotation_angle1 = 0
        self.rotation_angle2 = 0
        self.rotation_angle3 = 0
        self.randomize()

    def randomize(self):
        self.prob_rotate1 = random.uniform(0, 1)
        self.prob_rotate2 = random.uniform(0, 1)
        self.prob_rotate3 = random.uniform(0, 1)
        self.rotation_angle1 = random.randint(-20, 20)
        self.rotation_angle2 = random.randint(-20, 20)
        self.rotation_angle3 = random.randint(-20, 20)

    def transform(self, data, direction):
        data_aug = deepcopy(data)
        if direction == "forward":
            if self.prob_rotate1 > 0.5:
                data_aug = rotate(data_aug, self.rotation_angle1, axes=(1, 2), reshape=False)
            if self.prob_rotate2 > 0.5:
                data_aug = rotate(data_aug, self.rotation_angle2, axes=(1, 3), reshape=False)
            if self.prob_rotate3 > 0.5:
                data_aug = rotate(data_aug, self.rotation_angle3, axes=(2, 3), reshape=False)
        else:
            if self.prob_rotate3 > 0.5:
                data_aug = rotate(data_aug, -self.rotation_angle3, axes=(1, 2), reshape=False)
            if self.prob_rotate2 > 0.5:
                data_aug = rotate(data_aug, -self.rotation_angle2, axes=(0, 2), reshape=False)
            if self.prob_rotate1 > 0.5:
                data_aug = rotate(data_aug, -self.rotation_angle1, axes=(0, 1), reshape=False)
        return data_aug


class GammaContrast(Transform):
    def __init__(self):
        self.prob = 0
        self.gamma = 0
        self.randomize()

    def randomize(self):
        self.prob= random.uniform(0, 1)
        self.gamma = random.uniform(0.25, 1.75)

    def transform(self, data, direction):
        data_aug = deepcopy(data)
        if direction == "forward":
            if self.prob > 0.5:
                data_aug = np.power(data_aug, self.gamma)
        else:
            pass

        return data_aug

def generate_augmentations(modality: ImagingModalityType = ImagingModalityType.MRI) -> List[Transform]:
    """
    Creates and randomizes a list of transforms.

    Returns
    -------

    """
    aug_list = []
    fl = Flip()
    aug_list.append(fl)
    zo = Zoom()
    aug_list.append(zo)
    ro = Rotate()
    aug_list.append(ro)

    ## @TODO. Intensity-based augmentation cannot be performed here (i.e., after intensity normalization).
    # if modality == ImagingModalityType.MRI:
    #     gam = GammaContrast()
    #     aug_list.append(gam)

    return aug_list


def run_augmentations(aug_list: List[Transform], data: np.ndarray, direction: str) -> np.ndarray:
    """
    Runs all augmentations sequentially and returns an augmented version of the original input.

    Parameters
    ----------
    aug_list
    data
    direction

    Returns
    -------

    """
    data_aug = deepcopy(data)
    if direction == "forward":
        for l in aug_list:
            data_aug = l.transform(data_aug, direction)
    else:
        for l in reversed(aug_list):
            data_aug = l.transform(data_aug, direction)

    return data_aug
