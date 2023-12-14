# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from typing import Any, Callable, Optional, Tuple
import numpy as np

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import cv2
import torch

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img

def ROI_mask(image_h, image_w, text_polys, input_size):
    h, w = image_h, image_w
    fmap_h, fmap_w = input_size // 32, input_size // 32

    mask = np.zeros((h, w), dtype=np.uint8)

    for text_poly in text_polys:
        box_np = np.array(text_poly, dtype=np.int32)
        cv2.fillPoly(mask, [box_np], color=1)

    resized_mask = cv2.resize(mask, (fmap_h, fmap_w))
    torch_mask = torch.from_numpy(resized_mask).unsqueeze(0)

    return torch_mask

'''
Our own dataset
'''
class synthtext_dataset(Dataset):
    def __init__(self, data_folder, input_size, transforms):
        super(synthtext_dataset, self).__init__()
        self.img_path = data_folder + '/' + 'SynthText/'
        self.gt_path = data_folder + '/' + 'label/'
        self.input_size = input_size
        self.transform = transforms
        self.loader = pil_loader
        # get file names of each gt
        self.gt_files = [os.path.join(self.gt_path, gt_file)
                         for gt_file in sorted(os.listdir(self.gt_path))]
        print(len(self.gt_files))

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        gt_path = self.gt_files[index]
        foldernum, name = gt_path.split('--')
        imgname = name.replace('.txt', '.jpg')
        foldernum = re.findall(r'\b\d+\b', foldernum)[-1]

        text_polys = []
        #text_tags = []
        #last_words = []

        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                poly_lines = line.strip().split(',', 8)
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, poly_lines[:8]))
                #last_words.append(poly_lines[-1] + '\n')
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                '''
                if last_words == ' ' or last_words == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)                
                '''
        img = self.loader(str(self.img_path + foldernum + '/' + imgname))
        # Generate text mask before resize it
        text_mask = ROI_mask(img.size[0], img.size[1], text_polys, self.input_size)
        return self.transform(img), text_mask


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))




def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    trans_train = transforms.Compose([
        transforms.Resize(input_size, interpolation=interpolation),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    dataset_path = os.path.abspath(dataset_path)
    
    dataset_train, text_mask = synthtext_dataset(data_folder=dataset_path, transform=trans_train)
    print_transform(trans_train, '[pre-train]')
    return dataset_train, text_mask


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')