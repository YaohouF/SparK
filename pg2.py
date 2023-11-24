import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

class TextRegionMasker:
    def __init__(self, image_size=(512, 512), mask_ratio=0.6):
        self.image_size = image_size
        self.mask_ratio = mask_ratio

    def mask_text_regions(self, text_regions):
        # Create a binary mask for each text region and combine them
        indices = torch.zeros((1, 1, *self.image_size), dtype=torch.bool)

        for region in text_regions:
            # Convert the 4 points to a polygon mask
            polygon_mask = self.polygon_to_mask(region)

            # Update the indices tensor with the polygon mask
            indices |= polygon_mask

        # Apply mask ratio
        indices = self.apply_mask_ratio(indices)

        return indices

    def polygon_to_mask(self, polygon):
        polygon = [(int(point[0]), int(point[1])) for point in polygon]
        # Create a binary mask from the polygon
        mask = Image.new('L', self.image_size, 0)
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        return torch.tensor(np.array(mask), dtype=torch.bool).unsqueeze(0).unsqueeze(0)

    def apply_mask_ratio(self, indices):
        # Flatten the indices tensor
        flattened_indices = indices.view(-1)

        # Calculate the number of pixels to mask
        num_pixels_to_mask = int(self.mask_ratio * flattened_indices.size(0))

        # Randomly choose the indices to mask
        masked_indices = torch.randperm(flattened_indices.size(0))[:num_pixels_to_mask]

        # Update the selected indices to True
        flattened_indices[masked_indices] = True

        # Reshape the indices tensor to its original shape
        indices = flattened_indices.view(indices.size())

        return indices

# Example usage:
# Assuming you have an instance of TextRegionMasker named masker
image_size = (512, 512)
mask_ratio = 0.6
masker = TextRegionMasker(image_size=image_size, mask_ratio=mask_ratio)

text_regions = [
    [[277, 123], [330, 159], [311, 186], [259, 150]],
    [[81, 10], [90, 23], [35, 65], [25, 52]],
    [[125, 8], [140, 26], [78, 76], [63, 58]],
    [[518, 329], [586, 345], [579, 374], [511, 358]],
    [[520, 368], [594, 391], [586, 418], [512, 395]],
    [[3, 309], [56, 254], [130, 326], [77, 380]],
    [[61, 225], [117, 172], [220, 280], [164, 333]],
    [[127, 361], [255, 411], [231, 473], [103, 423]],
    [[120, 8], [219, 10], [218, 46], [120, 44]]
]

# Generate the text region mask
text_region_mask = masker.mask_text_regions(text_regions)

# Plot the mask on a blank image (for visualization purposes)
plt.imshow(text_region_mask.squeeze().cpu().numpy(), cmap='gray')
plt.show()



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
