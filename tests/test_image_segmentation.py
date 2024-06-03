import unittest
import numpy as np
from src.image_segmentation_model import ImageSegmentationModel
from src.logger import setup_logger

class TestImageSegmentationModel(unittest.TestCase):
    def setUp(self):
        self.logger = setup_logger('test_image_segmentation_model', 'logs/test_image_segmentation_model.log')
        self.logger.info('Setting up TestImageSegmentationModel...')
        self.model = ImageSegmentationModel()

    def test_build_model(self):
        self.logger.info('Running test_build_model...')
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model.input_shape[1:], self.model.input_shape)
        self.logger.info('test_build_model passed.')

    def test_preprocess_data(self):
        self.logger.info('Running test_preprocess_data...')
        annotation = {
            'size': [256, 256],
            'counts': np.array([256, 256])
        }
        mask = self.model.preprocess_data(annotation)
        self.assertEqual(mask.shape, (256, 256))
        self.logger.info('test_preprocess_data passed.')

    def test_train_and_predict(self):

        self.logger.info('Running test_train_and_predict...')
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
        self.logger.info('test_train_and_predict passed.')

if __name__ == "__main__":
    unittest.main()