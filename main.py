import torchvision.transforms as T
from src import LocalIterativeNeighborJitter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

# 1. Download a fresh sample from CIFAR-10
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=T.ToTensor()
)

# 2. SELECT 4 COMPLETELY RANDOM INDICES EVERY RUN
dataset_size = len(test_dataset)
random_indices = torch.randint(0, dataset_size, (4,)).tolist()
print(f"Selected random image IDs from dataset: {random_indices}")

# Pull the random samples and stack them into a clean batch [B=4, C=3, H=32, W=32]
xb_raw = torch.stack([test_dataset[idx][0] for idx in random_indices])

# --- THE FIX: KEEP A TRULY UNTOUCHED COPY OF THE ORIGINAL DATA ---
original_images = xb_raw.clone()

# 3. Instantiate the multi-channel augmentation class
# (We use higher settings here so you can easily spot where the texture patch landed)
color_jitter_fn = LocalIterativeNeighborJitter(
    num_iterations=250,        # Higher iterations to explicitly display the woven texture
    blend_rate=50,
    patch_size=150,            # Window size large enough to span across the color seam
    background_thresh=0.5,
)

# 4. Apply the augmentation to our raw batch
jittered_images = color_jitter_fn(xb_raw)


# Helper function to convert [C, H, W] Tensors to standard [H, W, C] images for plotting
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    return np.clip(image, 0, 1)


# 5. Render the pristine Original vs the Jittered result side-by-side
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for idx in range(4):
    # Top Row: Clean, untouched copies
    axes[0, idx].imshow(im_convert(original_images[idx]))
    axes[0, idx].set_title(f"Pristine Original {idx+1}")
    axes[0, idx].axis("off")

    # Bottom Row: Augmented images
    axes[1, idx].imshow(im_convert(jittered_images[idx]))
    axes[1, idx].set_title(f"Locally Jittered {idx+1}")
    axes[1, idx].axis("off")

plt.tight_layout()
plt.show()