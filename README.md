# Self-Supervised Metric Learning for Blood Cell Images

This repository is an attempt to implement a self-supervised learning model for blood cell images using with 2 approches:
- Triplet Loss
- Contrastive Loss

The model is trained on the [Blood Cell Images dataset](https://www.kaggle.com/paultimothymooney/blood-cells) from Kaggle. 
The dataset contains 12,500 images of blood cells, which are classified into 4 categories: eosinophil, lymphocyte, monocyte, and neutrophil.

However, since we want to train the model in a self-supervised manner, we will not use the labels provided in the dataset. 
Instead, we will use the triplet loss or contrastive loss to learn a feature representation of the images.

## Triple Loss Approach

```math
L(A, P, N) = max(0, d(A, P) - d(A, N) + margin)
```

Where:
- `A` is the anchor image
- `P` is the positive image (same class as anchor)
- `N` is the negative image (different class from anchor)
- `d(A, P)` is the distance between the anchor and positive images
- `d(A, N)` is the distance between the anchor and negative images
- `margin` is a hyperparameter that defines the minimum difference between the positive and negative distances

The goal of the model is to learn a feature representation of the images such that the distance between the anchor and positive images is minimized, while the distance between the anchor and negative images is maximized.

The negatives are found using batch hard negative mining, which selects the hardest negative for each anchor image.
This done by finding the negative image that is closest to the anchor image in terms of distance in the batch.

I decided to use batch hard negative mining because it is more computationally efficient then mining negatives from the entire dataset.

### Improvements
It is important to note that since we are in self-supervised learning, we may get some false negatives.


## Contrastive Loss Approach

```math
L(z_i, z_j) = -log \frac{exp(sim(z_i, z_j) / t)}{sum(exp(sim(z_i, z_k) / t))}
```

Where:
- `z_i` and `z_j` are the feature representations of the images
- `sim(z_i, z_j)` is the cosine similarity between the feature representations
- `N` is the number of images in the batch
- `t` is a temperature parameter that scales the similarity scores

The goal of the model is to learn a feature representation of the images such that the similarity between positive pairs is maximized, while the similarity between negative pairs is minimized.

## References

- [Triplet Loss](https://arxiv.org/abs/1503.03832)
- [Contrastive Loss](https://arxiv.org/abs/2002.05709)

---

## Requirements

The following packages are required to run the code provided in this repository:

- Python 3.10
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn