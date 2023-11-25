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

#get coords
text_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/label/1--ant+hill_1_0.txt'
text_polys = []
with open(text_path, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        poly_lines = line.strip().split(',', 8)
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, poly_lines[:8]))
        text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

def mask(B: int, device, image_w, image_h, generator=None, mask_ratio = 0.6):
    h, w = image_h, image_w
    fmap_h, fmap_w = h // 32, w // 32
    len_keep = round(fmap_h * fmap_w * (1 - mask_ratio))
    idx = torch.rand(B, fmap_h * fmap_w, generator=generator).argsort(dim=1)
    idx = idx[:, :len_keep].to(device)  # (B, len_keep)
    return torch.zeros(B, fmap_h * fmap_w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, fmap_h, fmap_w)

#算出长和宽去mask text region。
def text_mask(B: int, text_regions, device, generator=None, mask_ratio = 0.6):
    final_mask = torch.zeros(1, 1, 512, 512, dtype=torch.bool, device="cuda" if torch.cuda.is_available() else "cpu")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inp_bchw = torch.tensor(real_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
active_b1hw = text_mask(inp_bchw.shape[0], text_polys, inp_bchw.device)
print(active_b1hw.shape)





def plot_selected_position(image, selected_position, text_regions):
    # Convert the torch tensor to a NumPy array
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Convert the selected_position tensor to a NumPy array
    selected_position_np = selected_position.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Display the input image
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Input Image")

    # Display the text regions on the input image
    if text_regions:
        for text_region in text_regions:
            region_np = np.array(text_region)
            plt.plot(region_np[:, 0], region_np[:, 1], color='red', linewidth=2)

    # Display the input image
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.title("Input Image")

    # Display the selected position
    plt.subplot(1, 2, 2)
    plt.imshow(selected_position_np, cmap='gray')
    plt.title("Selected Position")

    plt.show()

plot_selected_position(inp_bchw, active_b1hw, text_polys)



'''
#active_b1ff = mask(inp_bchw.shape[0], inp_bchw.device) 
active_b1ff = mask(inp_bchw.shape[0], inp_bchw.device, 512, 512) 
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
'''
