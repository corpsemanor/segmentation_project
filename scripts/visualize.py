import cv2
import numpy as np
from src.image_segmentation_model import ImageSegmentationModel

def visualize(image_paths):
    model = ImageSegmentationModel()
    model.model.load_weights('unet_model_multiclass.keras')

    for image_path in image_paths:
        image = cv2.imread(image_path)
        predicted_mask = model.predict(image)
        
        num_classes = model.num_classes
        mask_color = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
        colors = np.random.randint(0, 255, size=(num_classes, 3))

        for class_id in range(num_classes):
            mask_color[predicted_mask == class_id] = colors[class_id]

        combined = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)

        cv2.imshow(f'Original Image - {image_path}', image)
        cv2.imshow(f'Predicted Mask - {image_path}', mask_color)
        cv2.imshow(f'Overlay - {image_path}', combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <path_to_image1> <path_to_image2> ...")
        sys.exit(1)

    image_paths = sys.argv[1:]
    visualize(image_paths)