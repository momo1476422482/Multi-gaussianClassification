from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from data_builder import ExtractPatches


class GreenData(Dataset):
    """
    A customized dataset used for training building from the DataBuilder
    """

    # ==============================================================================
    def __init__(
            self,

            input_path: Path,
            input_img_test: np.ndarray,
            output_path: Path,
            train: bool,
            patch_size: int = 15,
            gap: int = 1,
            rate: float = 1

    ):
        """
        for train phase :
        :param input_path: path of the folder where stock the images to be used for extraction of
               the patches for training
        :param output_path: path of the folder where to stock the patches to be used for training

        for inference phase :
        :param input_img_test: the test image

        common params :
        :param train: to indicate the phase of train/inference
        :param patch_size:
        :param gap:
        :param rate:
        """
        self.input_path = input_path
        self.output_path = output_path
        self.patch_size = patch_size
        self.gap = gap
        self.rate = rate
        self.input_img_test = input_img_test
        self.spliter_data = ExtractPatches(self.patch_size,
                                           basepath=self.input_path,
                                           input_img_test=self.input_img_test,
                                           outpath=self.output_path,
                                           thresh=0.7,
                                           gap=self.gap,
                                           save_mode='txt')

        self.train = train
        if self.train:
            self.patches, self.labels = self.spliter_data(self.rate, self.train)
        else:
            self.patches, self.coordinates = self.spliter_data(self.rate, self.train)

    # ==============================================================================
    def preprocessing(self, image: np.ndarray):

        return image / np.linalg.norm(image)

    # ==============================================================================
    def __len__(self):
        return len(self.patches)

    # ==============================================================================
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = self.patches[item]
        image = self.preprocessing(image)

        if self.train:
            label = self.labels[item]

        img_tensor = torch.from_numpy(image)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.float()
        '''

        if np.random.random_sample()<0.3:
            jitter = transforms.ColorJitter(brightness=.5, hue=.3)
            img_tensor = jitter(img_tensor)
        '''

        if self.train:
            return img_tensor, label
        else:
            return img_tensor, self.coordinates[item]


# ==================================================================
if __name__ == '__main__':
    patch_size = 11
    ftrain = GreenData(root=Path(__file__).parent,
                       train=True, patch_size=patch_size, gap=3)
    img, label = ftrain[0]

    print(img)
