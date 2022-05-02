import json
import random

import shapely.geometry as shgeo
from pathlib import Path
import numpy as np
import shapely.geometry as shgeo
from typing import List, Dict, Tuple
import cv2
import pandas as pd
from sklearn.feature_extraction import image
import time

# ============================================================================
from utils import LocalisedPatches


class ExtractPatches:
    """
    Prepare the dataset :
    for train : several images with labels
    for test: only one image given in input_path or ndarary
    """

    # ============================================================================
    def __init__(self, patch_size: int, basepath: Path,
                 outpath: Path,
                 input_img_test: np.ndarray,
                 gap: int = 56,
                 thresh: float = 0.98,
                 ext: str = ".png",
                 save_mode: str = 'txt') -> None:

        self.gap = gap
        self.patch_size = patch_size

        self.basepath = basepath
        self.outpath = outpath

        self.slide = self.patch_size - self.gap
        self.thresh = thresh
        self.ext = ext

        self.imagepath = self.basepath / "images"
        self.labelpath = self.basepath / "labelTxt"
        self.save_mode = save_mode
        self.list_categories = []
        self.input_img_test = input_img_test

        self.res_dict = {}

        if not self.outpath.is_dir():
            self.outpath.mkdir(parents=True, exist_ok=False)

    # ==================================================================================
    @staticmethod
    def resize_img(img: np.ndarray, rate: float) -> np.ndarray:
        if rate != 1:
            resizeimg = cv2.resize(
                img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        return resizeimg

    # ================================================================
    def create_output_path(self) -> None:
        """
        Creation of the directory of diffrent labeled patches with their corresponding annotation files
        :return:
        """

        paths_label = self.labelpath.glob('*' + self.save_mode)
        for path_label in paths_label:
            f = open(path_label, 'r')
            while True:
                line = f.readline()
                if line:
                    splitlines = line.strip().split(' ')
                    if len(splitlines) < 9:
                        continue
                    if len(splitlines) >= 9:
                        if not (self.outpath / splitlines[8]).is_dir():
                            (self.outpath / splitlines[8]).mkdir(parents=True, exist_ok=False)
                            self.res_dict.update({splitlines[8]: []})
                            self.list_categories.append(splitlines[8])
                else:
                    break

    # ================================================================
    @staticmethod
    def data2poly(path_label: Path) -> List[Dict]:

        """
                    parse the dota ground truth in the format:
                     [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                 """

        objects = []
        f = open(path_label, 'r')
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if len(splitlines) < 9:
                    continue
                if len(splitlines) >= 9:
                    object_struct['name'] = splitlines[8]
                if len(splitlines) == 9:
                    object_struct['difficult'] = '0'
                elif len(splitlines) >= 10:

                    object_struct['difficult'] = splitlines[9]
                object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                         (float(splitlines[2]), float(splitlines[3])),
                                         (float(splitlines[4]), float(splitlines[5])),
                                         (float(splitlines[6]), float(splitlines[7]))
                                         ]
                gtpoly = shgeo.Polygon(object_struct['poly'])
                object_struct['area'] = gtpoly.area

                objects.append(object_struct)
            else:
                break
        return objects

    # =====================================================================================
    @staticmethod
    def calculate_inter_iou(poly1: shgeo.Polygon, poly2: shgeo.Polygon) -> Tuple[shgeo.Polygon, float]:
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        inter_iou = inter_area / poly1_area
        return inter_poly, inter_iou

    # ====================================================================================
    def filter_patches(self, img: np.ndarray, subimgname: str, objects: List[Dict], left: float, up: float,
                       right: float,
                       down: float) -> None:
        """
            Get the list of patches within their corresponding labels
            :param subimgname:
            :param objects:
            :param left:
            :param up:
            :param right:
            :param down:
            :return:
            """

        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down), (left, down)])

        for obj in objects:

            gtpoly = shgeo.Polygon([(obj["poly"][0][0], obj["poly"][0][1]), (obj["poly"][1][0], obj["poly"][1][1]),
                                    (obj["poly"][2][0], obj["poly"][2][1]), (obj["poly"][3][0], obj["poly"][3][1]),
                                    ])

            if gtpoly.area <= 0:
                continue
            inter_poly, inter_iou = self.calculate_inter_iou(imgpoly, gtpoly)

            if inter_iou >= self.thresh:
                coord_poly_orig = list(shgeo.polygon.orient(inter_poly, sign=1).exterior.coords)[0:-1]
                if len(coord_poly_orig) == 4:
                    path_output = self.outpath / obj['name'] / str(subimgname + self.ext)
                    if (down - up) == self.patch_size and (right - left) == self.patch_size:
                        subimg = img[up: down, left: right]
                        cv2.imwrite(str(path_output), subimg)
                        self.res_dict[obj['name']].append(subimg)

    # =====================================================================================
    def label_patches(self) -> Tuple[List[np.ndarray], List[int]]:
        patches = []
        labels = []
        category_ix = 0
        classif_map = {}
        for c in self.list_categories:
            print(c, len(self.res_dict[c]))

        size = max([len(self.res_dict[c]) for c in self.list_categories])
        print(f"select {size} samples from {self.list_categories}")

        for category, s in self.res_dict.items():
            if category in self.list_categories:
                patches += s

                mul_size = int(size/len(s))
                if mul_size>1:
                    for ii in range(mul_size-1):
                        patches += s
                print('patches size',len(patches))


                classif_map.update({category: category_ix})
                labels += [category_ix] * mul_size*len(s)
                print('labels size', len(labels))
                category_ix += 1

        with open("classif_map.json", "w") as fp:
            json.dump(classif_map, fp, indent=4)
        return patches, labels

    # =====================================================================================
    def get_patches_train(self, name: str, extent: str, rate: float) -> None:
        """
            split a single image into patches
            :param name: image name
            :param rate: the resize scale for the image
            :param extent: the image format

            """
        img = cv2.imread(str(self.imagepath / str(name + extent)))
        # img=cv2.resize(img,(1920,1080))

        if np.shape(img) == ():
            return

        label_name = self.labelpath / str(name + ".txt")
        objects = self.data2poly(label_name)

        for obj in objects:
            # obj["poly"] = list(map(lambda x: rate * x, obj["poly"]))
            obj["poly"] = [(t[0] * rate, t[1] * rate) for t in obj["poly"]]

        resizeimg = self.resize_img(img, rate)
        outbasename = name + "__" + str(rate) + "__"
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while left < weight:
            if left + self.patch_size >= weight:
                left = max(weight - self.patch_size, 0)
            up = 0
            while up < height:
                if up + self.patch_size >= height:
                    up = max(height - self.patch_size, 0)
                right = min(left + self.patch_size, weight - 1)
                down = min(up + self.patch_size, height - 1)
                subimgname = outbasename + str(left) + "___" + str(up)

                self.filter_patches(resizeimg, subimgname, objects, left, up, right, down)
                if up + self.patch_size >= height:
                    break
                else:
                    up = up + self.slide
            if left + self.patch_size >= weight:
                break
            else:
                left = left + self.slide

    # =====================================================================================
    def get_patches_test(self, rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
            split a single image into patches
            :param name: image name
            :param rate: the resize scale for the image
            :param extent: the image format

            """
        patches = []
        coordinates = []
        # if (self.imagepath / str(name + extent)).is_file():
        # img = cv2.imread(str(self.imagepath / str(name + extent)))
        # else:
        img = self.input_img_test

        img = cv2.resize(img, (240, 135))

        if np.shape(img) == ():
            return patches, coordinates

        resizeimg = self.resize_img(img, rate)

        lp = LocalisedPatches(self.patch_size)
        patches, coordinates = lp.from_array(resizeimg)
        return patches, coordinates

    # =====================================================================================
    def __call__(self, rate: float, train: bool) -> Tuple[List, List]:
        pathlist = Path(self.imagepath).glob('**/*' + self.ext)
        if train is True:

            self.create_output_path()

            for path_data in pathlist:
                print(path_data)
                self.get_patches_train(path_data.stem, self.ext, rate)
            return self.label_patches()
        else:

            return self.get_patches_test(rate)


# ===============================================================
if __name__ == '__main__':
    filedir = Path(__file__).parent

    split = ExtractPatches(patch_size=55,
                           basepath=(filedir / "dataset" / "train"),
                           outpath=(filedir / "dataset" / "test_split1"),
                           thresh=0.7,
                           gap=5,
                           save_mode='txt')
    st = time.time()
    patches, labels = split(0.7, train=True)
    print(time.time() - st)
    print(len(patches), len(labels))
