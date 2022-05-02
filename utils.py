import json
from pathlib import Path
from typing import Tuple, cast

import cv2
import numpy as np
from sklearn.feature_extraction import image
import pandas as pd



class PatchExtactor:

    # ============================================================================
    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size

    # ====================================================
    def __call__(self, path: Path) -> np.ndarray:
        """
        Extract patches from the input image
        """

        assert path.is_file(), f"{path} does not exist"

        print(f"extract patches from {path}")

        return self.from_array(cv2.imread(str(path)))

    # ====================================================
    def from_array(self, array: np.ndarray) -> np.ndarray:
        print(f"extract patches from an image of dimension {array.shape}")

        return cast(
            np.ndarray,
            image.extract_patches_2d(array, (self.patch_size, self.patch_size)),
        )


# ============================================================================
class LocalisedPatches:

    # ============================================================================
    def __init__(self, patch_size: int) -> None:
        self.extractor = PatchExtactor(patch_size)

    # ============================================================================
    def set_identifiers(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dim_0 = image.shape[0]
        dim_1 = image.shape[1]

        dim_0_identifier = np.tile(np.arange(dim_0).reshape(-1, 1), (1, dim_1))
        dim_1_identifier = np.tile(np.arange(dim_1).reshape(1, -1), (dim_0, 1))

        return dim_0_identifier, dim_1_identifier

    # ============================================================================
    def __call__(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        assert path.is_file(), f"{path} does not exist"

        return self.from_array(cv2.imread(str(path)))

    # ============================================================================
    def from_array(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dim_0_identifier, dim_1_identifier = self.set_identifiers(image)

        patches = self.extractor.from_array(image)
        dim_0 = self.extractor.from_array(dim_0_identifier)
        dim_1 = self.extractor.from_array(dim_1_identifier)

        coordinates = np.stack(
            [np.array([p_0.min(), p_1.min()]) for p_0, p_1 in zip(dim_0, dim_1)]
        )

        return patches, coordinates


if __name__ == '__main__':
    path=Path(__file__)