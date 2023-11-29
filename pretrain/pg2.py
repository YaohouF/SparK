import os
from functools import partial
from typing import List
import re


import torch
import dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from utils.imagenet import synthtext_dataset

if __name__ == '__main__':
    # Your main code here
    data_path = "/Users/fanyaohou/Desktop/SynthText"
    input_size = 512
    dataset_train = synthtext_dataset(data_path, input_size)
    data_loader_train=DataLoader(
        dataset=dataset_train, num_workers=1, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=1,
            shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        ), worker_init_fn=worker_init_fn
    )
    
    itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)
    i = 0
    for j, obj in enumerate(data_loader_train):
        img, mask = obj
        i += 1
    # You can print or visualize the loaded data here
        print(f"Batch {j + 1} - Image shape: {img.shape}, mask shape: {mask.shape}")
        if i > 20:
            break
    # Add code for visualization if needed

'''
data_path = "/Users/fanyaohou/Desktop/SynthText"
img_path = data_path + '/' + 'SynthText/'
gt_path = data_path + '/' + 'label/'

gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
gt_path = gt_files[0]
foldernum, name = gt_path.split('--')
imgname = name.replace('.txt', '.jpg')
foldernum = re.findall(r'\b\d+\b', foldernum)[-1]
text_polys = []
with open(gt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            poly_lines = line.strip().split(',', 8)
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, poly_lines[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
img = cv2.imread(str(img_path + foldernum + '/' + imgname), cv2.IMREAD_COLOR)
img = cv2.resize(img, (512, 512))
img = torch.from_numpy(img.transpose(2, 0, 1)).float()

'''


