# Self-Supervised Learning with Triplet Loss for Blood Cell Images

This repository is an attempt to implement a self-supervised learning model for blood cell images using triplet loss. 

The model is trained on the [Blood Cell Images dataset](https://www.kaggle.com/paultimothymooney/blood-cells) from Kaggle. 
The dataset contains 12,500 images of blood cells, which are classified into 4 categories: eosinophil, lymphocyte, monocyte, and neutrophil.

However, since we want to train the model in a self-supervised manner, we will not use the labels provided in the dataset. 
Instead, we will use the triplet loss function to learn the features of the images.

The Triple Loss function is defined as follows:

```
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

I decided to use batch hard negative mining because it is computationally efficient then mining negatives from the entire dataset.

## Improvements
It is important to note that since we are in self-supervised learning, we may get some false negatives.
Another approach could be to use the Teacher-Student method, but I would need to do more research on that.

---

## Requirements

The following packages are required to run the code provided in this repository:

- Python 3.10
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn