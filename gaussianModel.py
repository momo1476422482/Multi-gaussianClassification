from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader

from dataset import GreenData


class gaussianModel:
    # =====================================================
    def __init__(self, num_patches: int, list_path_patches: List[Path], ext: str):
        """

        :param num_patches: number of patches used in train the multi-dimensional gaussian
        :param list_path_patches:
        :param ext:
        """
        self.num_patches = num_patches
        self.list_path_patches = list_path_patches
        self.ext = ext
        self.list_img_arrays = []
        self.list_dist = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_img_arrays()
        self.get_distributions()

    # =====================================================
    def get_img_arrays(self):

        for path_patches in self.list_path_patches:

            img_list = []
            for path_img in list(path_patches.glob('*.' + self.ext))[0:self.num_patches]:
                img = cv2.imread(str(path_img))
                img = img / np.linalg.norm(img)

                img_list.append(img.reshape(1, -1))

            img_array = np.array(img_list)
            img_array = img_array.reshape(img_array.shape[0] * img_array.shape[1], -1)

            self.list_img_arrays.append(img_array)
        '''
        img_array=np.random.randint(1,size=(20,30))
        '''

    # =====================================================
    @staticmethod
    def get_mean(input_array: np.ndarray):
        return np.mean(input_array, axis=0)
        # return np.array(input_array.mean())

    # =====================================================
    @staticmethod
    def get_cov(input_array: np.ndarray):
        return np.cov(input_array)
        # return np.std(input_array, axis=1)

    # =======================================================
    def get_distributions(self):
        for index, img_array in enumerate(self.list_img_arrays):
            gd = MultivariateNormal(torch.from_numpy(self.get_mean(img_array)).to(self.device),
                                    torch.from_numpy(
                                        self.get_cov(np.transpose(img_array)) + 0.01 * np.eye(img_array.shape[1])).to(
                                        self.device),
                                    )

            print(index, 'mean', self.get_mean(img_array), 'std', self.get_cov(np.transpose(img_array)))
            '''
            gd = Normal(torch.from_numpy(self.get_mean(img_array)).to(self.device),
                        torch.from_numpy(
                            self.get_cov(np.transpose(img_array))).to(
                            self.device),
                        )
            '''
            self.list_dist.append(gd)

    # =======================================================
    def __call__(self, patch_test: np.ndarray) -> Tensor:

        patch_test = patch_test.reshape(patch_test.shape[0],
                                        patch_test.shape[1] * patch_test.shape[2] * patch_test.shape[3])

        for index, dist in enumerate(self.list_dist):

            if index == 0:
                res_proba = dist.log_prob(patch_test).reshape(-1, 1)


            else:
                res_proba = torch.cat((res_proba, dist.log_prob(patch_test).reshape(-1, 1)), 1)

        return torch.argmax(res_proba, 1)


# ==========================================================
if __name__ == '__main__':
    patch_size = 7
    list_path_patches = []
    list_path_patches.append(Path(__file__).parent / 'dataset/train_split/background')
    list_path_patches.append(Path(__file__).parent / 'dataset/train_split/green')
    list_path_patches.append(Path(__file__).parent / 'dataset/train_split/brown')

    gm = gaussianModel(num_patches=1000, list_path_patches=list_path_patches, ext='png')

    dataset_test = GreenData(input_path=Path(__file__).parent / 'dataset/test',
                             output_path=Path(__file__).parent / 'dataset/train_split', rate=1,
                             train=False, patch_size=patch_size, gap=5,
                             input_img_test=cv2.imread('dataset/test/output_1_20-06-35_frame0.png'))

    test_dataloader = DataLoader(
        dataset_test,
        batch_size=15000,
    )

    res = []
    coordinates = []
    for iter_id, batch in enumerate(test_dataloader):
        data, coordinate = batch

        data = data.to(gm.device)
        output = gm(data)
        output_np = output.cpu().detach().numpy()

        if iter_id == 0:
            res = output_np
            coordinates = coordinate
        else:
            res = np.concatenate((res, output_np), axis=0)
            coordinates = np.concatenate((coordinates, coordinate), axis=0)

    mask = np.zeros(
        (270 - patch_size + 1, 480 - patch_size + 1)
    )
    mask[tuple(coordinates.T)] = res
    plt.figure()
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
