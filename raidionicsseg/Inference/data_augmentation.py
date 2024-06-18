import numpy as np
import random
import logging
from typing import List
from copy import deepcopy
from abc import ABC, abstractmethod
from scipy.ndimage import zoom, rotate


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
            if self.prob_zoom > 0.5:
                data_aug = zoom(data_aug, self.zoom_ratio)
        else:
            if self.prob_zoom > 0.5:
                data_aug = zoom(data_aug, 1 / self.zoom_ratio)
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
                data_aug = rotate(data_aug, self.rotation_angle1, axes=(0, 1), reshape=False)
            if self.prob_rotate2 > 0.5:
                data_aug = rotate(data_aug, self.rotation_angle2, axes=(0, 2), reshape=False)
            if self.prob_rotate3 > 0.5:
                data_aug = rotate(data_aug, self.rotation_angle3, axes=(1, 2), reshape=False)
        else:
            if self.prob_rotate3 > 0.5:
                data_aug = rotate(data_aug, -self.rotation_angle3, axes=(1, 2), reshape=False)
            if self.prob_rotate2 > 0.5:
                data_aug = rotate(data_aug, -self.rotation_angle2, axes=(0, 2), reshape=False)
            if self.prob_rotate1 > 0.5:
                data_aug = rotate(data_aug, -self.rotation_angle1, axes=(0, 1), reshape=False)
        return data_aug


def generate_augmentations() -> List[Transform]:
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
            data_aug = l.transform(data, direction)
    else:
        for l in reversed(aug_list):
            data_aug = l.transform(data, direction)

    return data_aug
