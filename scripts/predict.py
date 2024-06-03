import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import cv2
import numpy as np
from src.image_segmentation_model import ImageSegmentationModel

def main(image_path):
    model = ImageSegmentationModel()
    model.model.load_weights('unet_model_multiclass.keras')

    image = cv2.imread(image_path)
    predicted_mask = model.predict(image)

    # Preprocess the predicted mask
    num_classes = model.num_classes
    mask_color = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, size=(num_classes, 3))

    for class_id in range(num_classes):
        mask_color[predicted_mask == class_id] = colors[class_id]

    # Resize the mask to match the image size
    mask_color = cv2.resize(mask_color, image.shape[:2][::-1])

    # Ensure the mask has the same number of channels as the image
    if image.ndim == 3 and mask_color.ndim == 2:
        mask_color = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR)

    combined = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)

    cv2.imshow('Original Image', image)
    cv2.imshow('Predicted Mask', mask_color)
    cv2.imshow('Overlay', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)