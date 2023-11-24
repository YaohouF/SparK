import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import numpy as np

# Hypothetical Test Image
image_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/1/ant+hill_1_0.jpg'
test_image = cv2.imread(image_path)
real_image = cv2.resize(test_image, (512, 512))

#get coords
text_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/label/1--ant+hill_1_0.txt'
text_polys = []

with open(text_path, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        poly_lines = line.strip().split(',', 8)
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, poly_lines[:8]))
        text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    east_polys = np.array(text_polys, dtype=np.float32)

print(text_polys)
for bbox_points in text_polys:
    x_coords, y_coords = zip(*bbox_points)
    print(x_coords)
# Convert to PyTorch tensor
inp_bchw = torch.tensor(real_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

# Masker Class

def mask(B: int, device, generator=None, mask_ratio = 0.6):
    h, w = 512, 512
    fmap_h, fmap_w = 512 // 32, 512 // 32
    len_keep = round(fmap_h * fmap_w * (1 - mask_ratio))
    idx = torch.rand(B, fmap_h * fmap_w, generator=generator).argsort(dim=1)
    idx = idx[:, :len_keep].to(device)  # (B, len_keep)
    return torch.zeros(B, fmap_h * fmap_w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, fmap_h, fmap_w)

def text_only_mask(B: int, device, text_regions, generator=None, mask_ratio = 0.6):
    h, w = 512, 512
    fmap_h, fmap_w = 512 // 32, 512 // 32
    len_keep = round(fmap_h * fmap_w * (1 - mask_ratio))

    mask = torch.zeros(B, 1, fmap_h * fmap_w, dtype=torch.bool, device=device)
    print(mask.shape)
    for bbox_points in text_regions:
        x_coords, y_coords = zip(*bbox_points)
        # Calculate the indices corresponding to the text region
        indices = (
            slice(None),
            slice(None),
            slice(min(y_coords) // 32, max(y_coords) // 32),
            slice(min(x_coords) // 32, max(x_coords) // 32)
        )
        print(indices)
        mask[indices] = True
    
    return mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the Masker class
active_b1ff = text_only_mask(inp_bchw.shape[0], inp_bchw.device, text_polys) 
print(active_b1ff.shape)
active_b1hw = active_b1ff.repeat_interleave(32, 2).repeat_interleave(32, 3)
print(active_b1hw.shape)
active_b1hw = active_b1hw.view(1,1,512,512)
# Element-wise multiplication to apply the mask
masked_bchw = inp_bchw * active_b1hw

# Convert masked_bchw back to numpy array for visualization
masked_image = masked_bchw.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Visualize the original and masked images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(real_image[:, :, ::-1])  # Convert BGR to RGB for display
axs[0].set_title('Original Scene Text Image')
axs[0].axis('off')

axs[1].imshow(masked_image)
axs[1].set_title('Masked Scene Text Image')
axs[1].axis('off')

plt.show()