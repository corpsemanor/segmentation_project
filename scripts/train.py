import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import tensorflow_datasets as tfds
from src.image_segmentation_model import ImageSegmentationModel

def main():
    model = ImageSegmentationModel()

    dataset = model.load_dataset()
    train_images, train_masks = model.prepare_data(dataset)
    val_images, val_masks = model.prepare_data(dataset)

    train_images, train_masks = train_images[:2000], train_masks[:2000]
    val_images, val_masks = val_images[4000:], val_masks[4000:]
    
    history = model.train(train_images, train_masks, val_images, val_masks)

if __name__ == "__main__":
    main()