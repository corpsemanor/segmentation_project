import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow_datasets as tfds
from src.image_segmentation_model import ImageSegmentationModel
from src.logger import setup_logger
import numpy as np

def main():
    logger = setup_logger('train_script', 'logs/train.log')
    logger.info('Starting training script...')

    model = ImageSegmentationModel()

    dataset = model.load_dataset()
    images, masks = model.prepare_data(dataset)

    train_images, train_masks = images[:2000], masks[:2000]
    val_images, val_masks = images[4400:], masks[4400:]
    pred_images = train_images[-5:]
    model.train(train_images, train_masks, val_images, val_masks)

    logger.info('Prediction script completed.')
    logger.info('Training script completed.')

if __name__ == "__main__":
    main()