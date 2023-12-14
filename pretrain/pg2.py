import os
from functools import partial
from typing import List
import re
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms

import PIL.Image as PImage
import torch

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC

#Use original function
trans_train = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.67, 1.0), interpolation=interpolation),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img

img = pil_loader('/Users/fanyaohou/Desktop/SynthText/SynthText/172/swan_2_6.jpg')
print(img)
#img = trans_train(img)



# Use our function
transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
img2 = cv2.imread('/Users/fanyaohou/Desktop/SynthText/SynthText/172/swan_2_6.jpg', cv2.IMREAD_COLOR)
img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
print(img2.size)

