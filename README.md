
# Image Segmentation Model

This project implements an image segmentation model using a modified U-Net architecture. The model is designed to predict segmentation masks for a given set of images.


## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Training

To train the model, run:

```
python scripts/train.py
```

## Prediction

To make a prediction on an image, run:

```
python scripts/predict.py <path_to_image>
```

## Testing

To run the unit tests, run:

```
python -m unittest discover tests
```
