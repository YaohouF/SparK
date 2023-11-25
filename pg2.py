import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import cv2
import numpy as np

# Extract input data, image and bounding boxes
image_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/1/ant+hill_1_0.jpg'
test_image = cv2.imread(image_path)
real_image = cv2.resize(test_image, (512, 512))
text_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/label/1--ant+hill_1_0.txt'
text_polys = []

with open(text_path, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        poly_lines = line.strip().split(',', 8)
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, poly_lines[:8]))
        text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


#mask = np.zeros((512, 512), dtype=np.uint8)

def mask_patch(patch, mask_ratio):
    return np.random.choice([0, 255], size=patch.shape, p=[1 - mask_ratio, mask_ratio])


# Create a mask for the bounding box
for text_region in text_polys:
    text_region_array = np.array(text_region, dtype=np.int32).reshape((4, 2))
    for i in range(0, len(text_region_array), 2):
        x1, y1, x2, y2 = text_region_array[i:i + 2].flatten()
        patch = real_image[y1:y2, x1:x2]
        masked_patch = mask_patch(patch, 0.7)
        real_image[y1:y2, x1:x2] = masked_patch

cv2.imshow("Result Image", real_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
image_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/1/ant+hill_1_0.jpg'
input_image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
input_image = transform(input_image).unsqueeze(0)
# Generate the text region mask
text_region_mask = masker.mask_text_regions(B=1, device='cpu', text_regions=text_regions)
active_b1hw = text_region_mask.repeat_interleave(32, 2).repeat_interleave(32, 3)
masker.plot_mask_on_image(input_image, active_b1hw)


'''
