import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

# Hypothetical Test Image
image_path = '/Users/fanyaohou/Desktop/SynthText/SynthText/1/ant+hill_1_0.jpg'
test_image = cv2.imread(image_path)
# Convert BGR to RGB
real_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
real_image_rgb = cv2.resize(real_image_rgb, (512, 512))

# Convert to PyTorch tensor
real_image_tensor = ToTensor()(real_image_rgb).unsqueeze(0)  # Add batch dimension

# Masker Class

def mask(B: int, device, generator=None, mask_ratio = 0.6):
    h, w = 512, 512
    len_keep = round(h * w * (1 - mask_ratio))
    idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
    idx = idx[:, :len_keep].to(device)  # (B, len_keep)
    return torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the Masker class
active_b1ff: torch.BoolTensor = mask(B=1, device=device)
active_b1hw = active_b1ff.repeat_interleave(32, 2).repeat_interleave(32, 3)  # (B, 1, H, W)
# Apply Mask to the Real Image

masked_real_image = real_image_tensor * active_b1hw.float()

# Display the Original and Masked Real Images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(real_image_rgb)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(masked_real_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
axs[1].set_title('Masked Image')
axs[1].axis('off')

plt.show()