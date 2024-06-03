import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
from keras import layers, models, applications
from pycocotools import mask as maskUtils

class ImageSegmentationModel:
    def __init__(self, input_shape=(256, 256, 3), num_classes=11):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        base_model = applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        base_model.trainable = True

        for layer in base_model.layers[:20]:
            layer.trainable = False

        conv1 = base_model.get_layer('block2a_expand_activation').output
        conv2 = base_model.get_layer('block3a_expand_activation').output
        conv3 = base_model.get_layer('block4a_expand_activation').output
        conv4 = base_model.get_layer('block6a_expand_activation').output
        conv5 = base_model.get_layer('top_activation').output

        up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
        up6 = layers.concatenate([up6, conv4], axis=3)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
        up7 = layers.concatenate([up7, conv3], axis=3)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
        up8 = layers.concatenate([up8, conv2], axis=3)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
        up9 = layers.concatenate([up9, conv1], axis=3)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

        up10 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv9)
        conv10 = layers.Conv2D(32, 3, activation='relu', padding='same')(up10)
        conv10 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv10)

        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(conv10)

        model = models.Model(inputs=[inputs], outputs=[outputs])

        return model

    def preprocess_data(self, annotation):
        size = annotation['size']
        mask_encoded = annotation['counts']
        mask_encoded = np.array2string(annotation['counts'].numpy())
        mask_encoded = mask_encoded[3:-2]
        rle = {'size': size, 'counts': mask_encoded.replace('\\\\','\\')}
        mask = maskUtils.decode(rle)
        return mask

    def load_dataset(self, dataset_name='test_coco_dataset', split='train'):
        return tfds.load(dataset_name, split=split, with_info=False)
    
    def prepare_data(self, dataset):
        images = []
        image_id_to_mask = {}
        image_id_to_index = {}

        for idx, example in enumerate(dataset):
            img_id = example['image_id'].numpy()
            image = example['image'].numpy()
            image = cv2.resize(image, (256, 256))
            images.append(image)
            image_id_to_index[img_id] = idx

        images = np.array(images)
        masks = np.zeros((len(images), 256, 256), dtype=np.uint8)

        for example in dataset:
            img_id = example['image_id'].numpy()
            segmentation = example['segmentation']
            object_class = example['label'].numpy()
            mask = self.preprocess_data(segmentation)
            mask = cv2.resize(mask, (256, 256))
            mask[mask == 1] = object_class

            if img_id in image_id_to_mask:
                existing_mask = image_id_to_mask[img_id]
                combined_mask = np.maximum(existing_mask, mask)
                image_id_to_mask[img_id] = combined_mask
            else:
                image_id_to_mask[img_id] = mask

        masks = [image_id_to_mask[example['image_id'].numpy()] for example in dataset]

        masks = np.array(masks).astype(np.float32)
        images = images / 255.0

        return images, masks

    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1.):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        return 1. - score

    def dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tf.keras.backend.flatten(tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int32'), num_classes=self.num_classes))
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    def data_generator(self, images, masks, batch_size):
        while True:
            for start in range(0, len(images), batch_size):
                end = min(start + batch_size, len(images))
                batch_images = images[start:end]
                batch_masks = masks[start:end]

                batch_images = np.array(batch_images)
                batch_masks = np.array(batch_masks)

                yield batch_images, batch_masks

    def train(self, train_images, train_masks, val_images, val_masks, epochs=75, batch_size=16):
        model = self.model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', self.dice_coefficient])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_model_multiclass.keras', save_best_only=True, monitor='val_loss', mode='min')

        train_gen = self.data_generator(train_images, train_masks, batch_size)
        val_gen = self.data_generator(val_images, val_masks, batch_size)

        num_train_images = len(train_images) if isinstance(train_images, list) else train_images.shape[0]
        num_val_images = len(val_images) if isinstance(val_images, list) else val_images.shape[0]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=num_train_images // batch_size,
            validation_steps=num_val_images // batch_size,
            callbacks=[early_stopping, model_checkpoint]
        )

        return history

    def predict(self, image):
        image = cv2.resize(image, (256, 256))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = self.model.predict(image)
        predicted_mask = np.argmax(prediction, axis=-1)
        predicted_mask = predicted_mask[0]

        return predicted_mask