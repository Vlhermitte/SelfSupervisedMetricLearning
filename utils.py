import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def data_augmentation(img):
    """
    Data augmentation function
    :param img: the input image
    :return: the augmented image
    """
    # Perform data augmentation here
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
    ])
    if img.device.type == 'mps': # mps not supported. See https://github.com/pytorch/pytorch/issues/77764
        augmented_img = transform(img.cpu()).to(img.device)
    else:
        augmented_img = transform(img)
    return augmented_img

def mine_hard_negatives(model, batch):
    """
    Mine hard negatives for the triplet loss, in a self-supervised setting
    Anchor: a sample from the batch
    Positive: augmented version of the anchor
    Negative: hardest sample from the batch
    :param model: the model
    :param batch: the batch
    :return: the hard negatives
    """
    # Apply data augmentation to get the positives
    augmented_batch = torch.stack([data_augmentation(img) for img in batch])

    # Combine original and augmented batches
    combined_batch = torch.cat([batch, augmented_batch], dim=0)

    # Set the model to evaluation mode
    model.eval()
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Forward pass to get embeddings
        embeddings = model(combined_batch)

    # Return the model to training mode
    model.train()
    batch_size = embeddings.size(0) // 2  # Since combined_batch contains both original and augmented images

    anchors = embeddings[:batch_size]
    positives = embeddings[batch_size:]

    # Compute distances between anchors and all other samples
    distances = torch.cdist(anchors, embeddings, p=2)  # Euclidean distance

    # Set distances of anchors to themselves and their positives to a large value (to exclude them)
    for i in range(batch_size):
        distances[i, i] = float('inf')
        distances[i, batch_size + i] = float('inf')

    # Hard negatives are the closest (smallest distance) samples to each anchor
    hard_negatives_indices = torch.argmin(distances, dim=1)
    hard_negatives = combined_batch[hard_negatives_indices]

    return batch, augmented_batch, hard_negatives


