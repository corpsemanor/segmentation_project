import numpy as np
import mlflow
from keras import callbacks
import matplotlib.pyplot as plt
import os

class MLflowCallback(callbacks.Callback):
    def __init__(self, val_data, num_images=5):
        super().__init__()
        self.val_data = val_data
        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_masks = self.val_data
        predictions = self.model.predict(val_images[:self.num_images])
        self.log_images(val_images, val_masks, predictions, epoch)

    def log_images(self, images, masks, predictions, epoch):
        num_images = min(len(images), self.num_images)
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
        for i in range(num_images):
            axes[i][0].imshow(images[i])
            axes[i][0].set_title('Image')
            axes[i][0].axis('off')
            
            # Check if the mask has 3 dimensions
            if masks[i].ndim == 3:
                axes[i][1].imshow(masks[i][:, :, 0], cmap='gray')
            else:
                axes[i][1].imshow(masks[i], cmap='gray')
            axes[i][1].set_title('True Mask')
            axes[i][1].axis('off')

            prediction_mask = np.argmax(predictions[i], axis=-1)
            axes[i][2].imshow(prediction_mask, cmap='gray')
            axes[i][2].set_title('Predicted Mask')
            axes[i][2].axis('off')
        save_dir = 'save'
        save_path = os.path.join(save_dir, f"epoch_{epoch + 1}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        # mlflow.log_artifact(f"epoch_{epoch + 1}.png")
