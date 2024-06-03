import unittest
import numpy as np
from src.image_segmentation_model import ImageSegmentationModel

class TestImageSegmentationModel(unittest.TestCase):
    def setUp(self):
        self.model = ImageSegmentationModel()

    def test_build_model(self):
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model.input_shape[1:], self.model.input_shape)

    def test_preprocess_data(self):
        annotation = {
            'size': [256, 256],
            'counts': np.array([256, 256])
        }
        mask = self.model.preprocess_data(annotation)
        self.assertEqual(mask.shape, (256, 256))

    def test_train_and_predict(self):
        # Fake data for testing
        train_images = np.random.rand(10, 256, 256, 3)
        train_masks = np.random.randint(0, 11, (10, 256, 256))
        val_images = np.random.rand(5, 256, 256, 3)
        val_masks = np.random.randint(0, 11, (5, 256, 256))

        history = self.model.train(train_images, train_masks, val_images, val_masks, epochs=50, batch_size=2)

        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)

        test_image = np.random.rand(256, 256, 3)
        predicted_mask = self.model.predict(test_image)
        self.assertEqual(predicted_mask.shape, (256, 256))

if __name__ == "__main__":
    unittest.main()