import torch
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import numpy as np

# Hypothetical Test Image
image_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/1/ant+hill_1_0.jpg'
test_image = cv2.imread(image_path)
real_image = cv2.resize(test_image, (512, 512))
real_image = real_image/255.0
device='cuda' if torch.cuda.is_available() else 'cpu'

#get coords
text_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/label/1--ant+hill_1_0.txt'
text_polys = []
with open(text_path, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        poly_lines = line.strip().split(',', 8)
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, poly_lines[:8]))
        text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

def mask(B: int, device, image_h, image_w, generator=None, mask_ratio = 0.6):
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
    fmap_h, fmap_w = h // 32, w // 32
    masks = []
    for i in range(B):
        mask = np.zeros((fmap_h, fmap_w), dtype=np.uint8)
        text_polys = text_polys_list[i]

        for text_poly in text_polys:
            box_np = np.array(text_poly, dtype=np.int32)
            cv2.fillPoly(mask, [box_np], color=1)

        resized_mask = cv2.resize(mask, (fmap_h, fmap_w))
        torch_mask = torch.from_numpy(resized_mask).unsqueeze(0).to(device)
        masks.append(torch_mask)

    return torch.stack(masks)

# get random mask
image_size = (real_image.shape[0], real_image.shape[1])
random_b1ff = mask(1, device,real_image.shape[0], real_image.shape[1])
print(random_b1ff.shape)

# get text mask
text_polys_list = [text_polys]
text_b1ff = ROI_mask(1, device,real_image.shape[0], real_image.shape[1], text_polys_list)
print(text_b1ff.shape)

active_b1hw = random_b1ff*text_b1ff
active_b1hw = active_b1hw.repeat_interleave(32, 2).repeat_interleave(32, 3)

masked_image = real_image.copy()
masked_image[active_b1hw > 0] = [255, 0, 0]
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(real_image)
axs[0].set_title('Original Image')

axs[1].imshow(masked_image)
axs[1].set_title('Masked Image')
plt.show()



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

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inp_bchw = torch.tensor(real_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
active_b1hw = text_mask(inp_bchw.shape[0], text_polys, inp_bchw.device)
print(active_b1hw.shape)
'''






