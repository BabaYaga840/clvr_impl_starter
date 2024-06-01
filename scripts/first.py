import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from general_utils import AttrDict
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import make_image_seq_strip
from sprites_datagen.rewards import ZeroReward

# Assuming MovingSpriteDataset and necessary classes are defined/imported above

# Setup the spec configuration for the dataset
spec = AttrDict(
        resolution=128,
        max_seq_len=30,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=4,      # number of shapes per trajectory
        rewards=[ZeroReward],
    )

# Initialize the dataset
dataset = MovingSpriteDataset(spec)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Function to visualize the images
def show_all_sprites(batch_images):
    """
    Visualizes all frames for each sequence in the batch.
    Args:
        batch_images (numpy array): Batch of images with shape (batch_size, seq_len, channels, height, width).
    """
    num_sequences, seq_len, channels, height, width = batch_images.shape
    
    fig, axes = plt.subplots(num_sequences, seq_len, figsize=(2 * seq_len, 2 * num_sequences))
    for i in range(num_sequences):
        for j in range(seq_len):
            ax = axes[i][j]
            # Normalize and permute the dimensions from (C, H, W) to (H, W, C) for plotting
            image = batch_images[i, j].transpose(1, 2, 0) / 255.0
            ax.imshow(image)
            ax.axis('off')
    plt.show()

def show_sprites(batch_images, num_sequences=4):
    """
    Visualizes the first frame of each sequence in the batch.
    Args:
        batch_images (numpy array): Batch of images with shape (batch_size, seq_len, channels, height, width).
        num_sequences (int): Number of sequences to visualize from the batch.
    """
    first_frames = batch_images[:, 15]  # This slices out the first frame from each sequence in the batch

    first_frames_tensor = torch.from_numpy(first_frames).float() / 255.0

    grid = make_grid(first_frames_tensor, nrow=num_sequences)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()

# Iterate over the DataLoader and visualize one batch
for data in dataloader:
    images = data.images.numpy()  # Converting images from torch tensor to numpy array for visualization
    print(images.shape)
    show_sprites(images)
    """gen = DistractorTemplateMovingSpritesGenerator(spec)
    traj = gen.gen_trajectory()
    img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
    cv2.imwrite("test.png", img[0].transpose(1, 2, 0))"""
    break  # We only show one batch for demonstration
