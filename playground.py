import torch
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import numpy as np

def resize_padding(image, w=512):
    max_wh = max(image.shape[0], image.shape[1])
    newImage = np.zeros((max_wh, max_wh, 3), np.uint8)
    newImage[:image.shape[0], :image.shape[1], :] = image
    newImage = cv2.resize(newImage, (w, w))
    return newImage

# Hypothetical Test Image
image_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/188/village_9_0.jpg'
test_image = cv2.imread(image_path)
real_image = cv2.resize(test_image, (576, 576))
real_image = real_image/255.0
device='cuda' if torch.cuda.is_available() else 'cpu'

#get coords
text_path = '/Users/fanyaohou/Desktop/SynthText/label/188--village_9_0.txt'
text_polys = []
with open(text_path, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        poly_lines = line.strip().split(',', 8)
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, poly_lines[:8]))
        text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

def mask(B: int, device, image_h, image_w, generator=None, mask_ratio = 0.4):
    h, w = image_h, image_w
    fmap_h, fmap_w = h // 32, w // 32
    len_keep = round(fmap_h * fmap_w * (1 - mask_ratio))
    idx = torch.rand(B, fmap_h * fmap_w, generator=generator).argsort(dim=1)
    idx = idx[:, :len_keep].to(device)  # (B, len_keep)
    return torch.zeros(B, fmap_h * fmap_w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, fmap_h, fmap_w)

'''
def mask_one(image_w, image_h, generator=None, mask_ratio = 0.4):
    h, w = image_h, image_w
    fmap_h, fmap_w = h // 32, w // 32
    len_keep = round(fmap_h * fmap_w * (1 - mask_ratio))
    idx = torch.rand(1, fmap_h * fmap_w, generator=generator).argsort(dim=1)
    idx = idx[:, :len_keep].to('cpu')  # (B, len_keep)
    return torch.zeros(1, fmap_h * fmap_w, dtype=torch.bool).scatter_(dim=1, index=idx, value=True).view(fmap_h, fmap_w)
'''

def ROI_mask(B: int, device, image_h, image_w, text_polys_list):
    h, w = image_h, image_w
    fmap_h, fmap_w = 576 // 32, 576 // 32
    masks = []
    for i in range(B):
        mask = np.zeros((h, w), dtype=np.uint8)
        text_polys = text_polys_list[i]

        for text_poly in text_polys:
            box_np = np.array(text_poly, dtype=np.int32)
            cv2.fillPoly(mask, [box_np], color=1)

        resized_mask = cv2.resize(mask, (fmap_h, fmap_w))
        torch_mask = torch.from_numpy(resized_mask).unsqueeze(0).to(device)
        masks.append(torch_mask)

    return torch.stack(masks)


#-------------------------------------------------------------------------
# This piece of code use bbox to scale up the probability of text region to be masked.
# scaling_factor [0, 1]
def scale_mask(B: int, device, image_h, image_w, mask_image, scaling_factor=0, generator=None, mask_ratio=0.6):
    h, w = image_h, image_w
    fmap_h, fmap_w = h // 32, w // 32
    len_keep = round(fmap_h * fmap_w * (1 - mask_ratio))
    
    # Generate random indices and normalize it
    random_idx = torch.rand(B, fmap_h * fmap_w, generator=generator)
    random_idx = random_idx / random_idx.max()

    scaled_values = scaling_factor * (mask_image.view(B, fmap_h * fmap_w) + 0.001)
    random_idx += scaled_values

    idx = random_idx.argsort(dim=1)[:, :len_keep].to(device)  # (B, len_keep)
    
    return torch.zeros(B, fmap_h * fmap_w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, fmap_h, fmap_w)


text_polys_list = [text_polys]
text_b1ff = ROI_mask(1, device,test_image.shape[0], test_image.shape[1], text_polys_list)
print(text_b1ff)
image_size = (real_image.shape[0], real_image.shape[1])
active_b1ff = scale_mask(1, device,real_image.shape[0], real_image.shape[1], text_b1ff, scaling_factor=0.1)
active_b1hw = active_b1ff.repeat_interleave(32, 2).repeat_interleave(32, 3)
inp_bchw = torch.tensor(real_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
masked_bchw = inp_bchw*active_b1hw

plt.subplot(1, 4, 1)
plt.imshow(text_b1ff[0, 0].cpu().numpy(), cmap='gray')
plt.title('text Mask')

# Visualize text mask
plt.subplot(1, 4, 2)
plt.imshow(active_b1ff[0, 0].cpu().numpy(), cmap='gray')
plt.title('combine Mask 16')

plt.subplot(1, 4, 4)
plt.imshow(masked_bchw[0, 0].cpu().numpy(), cmap='gray')
plt.title('real image')
plt.show()



'''
This piece of code use the original version to mask the image
bbox as attention map * 'mask' to generate mask image only for text region
# get random mask
image_size = (real_image.shape[0], real_image.shape[1])
random_b1ff = mask(1, device,real_image.shape[0], real_image.shape[1])

# get text mask
text_polys_list = [text_polys]
text_b1ff = ROI_mask(1, device,test_image.shape[0], test_image.shape[1], text_polys_list)

active_b1hw = random_b1ff*text_b1ff ^ 1
active_b1hw = active_b1hw == 1
print(active_b1hw)
active_b1hw = active_b1hw.repeat_interleave(32, 2).repeat_interleave(32, 3)

inp_bchw = torch.tensor(real_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
masked_bchw = inp_bchw*active_b1hw
print(masked_bchw.shape)

# Visualize random mask
plt.subplot(1, 4, 1)
plt.imshow(random_b1ff[0, 0].cpu().numpy(), cmap='gray')
plt.title('Random Mask')

# Visualize text mask
plt.subplot(1, 4, 2)
plt.imshow(text_b1ff[0, 0].cpu().numpy(), cmap='gray')
plt.title('Text Mask')

# Visualize combined mask
plt.subplot(1, 4, 3)
plt.imshow(active_b1hw[0, 0].cpu().numpy(), cmap='gray')
plt.title('Combined Mask')

plt.subplot(1, 4, 4)
plt.imshow(masked_bchw[0, 0].cpu().numpy(), cmap='brg')
plt.title('real image')

plt.show()
'''


#算出长和宽去mask text region。
#这里想要用的话想要找到大的矩形框去覆盖原来的rotated box。
#然后用原版mask function生成mask再贴回原图
def text_mask(B: int, text_regions, device, generator=None, mask_ratio = 0.6):
    final_mask = torch.zeros(B, 1, 512, 512, dtype=torch.bool, device="cuda" if torch.cuda.is_available() else "cpu")
    i = 0
    for text_region in text_regions:
        i += 1
        [x1, y1], [x2, y2], [x3, y3], [x4, y4] = text_region
        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)

        width = max_x - min_x
        height = max_y - min_y
        patch_mask = mask(B, device, width, height)
        region_mask = patch_mask.repeat_interleave(32, 2).repeat_interleave(32, 3)
        resized_region_mask = F.interpolate(region_mask.float(), size=(height, width), mode='nearest').bool()
        print(resized_region_mask.shape)
        print(final_mask[:, :, min_y:max_y, min_x:max_x].shape, min_x-max_x)
        final_mask[:, :, min_y:max_y, min_x:max_x] |= resized_region_mask
        if i == 3:
            return final_mask

    return final_mask







