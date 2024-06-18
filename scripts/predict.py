import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from src.logger import setup_logger
import matplotlib.pyplot as plt
import tensorflow as tf

def main(image_path):

    logger = setup_logger('predict_script', 'logs/predict.log')
    logger.info(f'Starting prediction script for image: {image_path}')

    model = tf.keras.models.load_model('saved_model.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    image = cv2.imread(image_path)

    if image is None:
        logger.error(f'Failed to load image at {image_path}')
        sys.exit(1)

    show_img = cv2.resize(image, (256, 256))
    image = cv2.resize(image, (256, 256)) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    prediction_mask = np.argmax(prediction, axis=-1)[0]

    visualize_prediction(show_img, prediction_mask)

    logger.info('Prediction script completed.')

def visualize_prediction(image, prediction):
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image)


    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(prediction, cmap='jet', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)