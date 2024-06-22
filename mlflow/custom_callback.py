import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from keras import callbacks
from src.image_segmentation_model import ImageSegmentationModel
from callback import MLflowCallback

os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''

os.environ['MLFLOW_TRACKING_USERNAME'] = ''
os.environ['MLFLOW_TRACKING_PASSWORD'] = ''

mlflow.set_tracking_uri("")
mlflow.set_experiment("vladimir_semantic")

try:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("vladimir_semantic")

    if experiment:
        print(f"Experiment '{experiment.name}' exists with ID {experiment.experiment_id}")
    else:
        print(f"Experiment 'vladimir_semantic' does not exist. Creating new experiment.")
        experiment_id = mlflow.create_experiment("vladimir_semantic")
        client.create_experiment_permission(experiment_id, 'mlflow', 'MANAGE')

except Exception as e:
    print(f"Failed to connect to MLflow server: {e}")
    sys.exit(1)

input_shape = (256, 256, 3)
num_classes = 11
image_segmentation_model = ImageSegmentationModel(input_shape=input_shape, num_classes=num_classes)
model = image_segmentation_model.model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

dataset = image_segmentation_model.load_dataset()
images, masks = image_segmentation_model.prepare_data(dataset)

train_images, train_masks = images[:2000], masks[:2000]
val_images, val_masks = images[4400:], masks[4400:]

batch_size = 16
train_generator = image_segmentation_model.data_generator(train_images, train_masks, batch_size)
val_generator = image_segmentation_model.data_generator(val_images, val_masks, batch_size)

class CustomModelCheckpoint(callbacks.Callback):
    def __init__(self, filepath, save_freq=5):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            filepath = self.filepath.format(epoch=epoch + 1)
            self.model.save(filepath)
            mlflow.log_artifact(filepath)

checkpoint_path = "checkpoints/epoch_{epoch:02d}.h5"
checkpoint_callback = CustomModelCheckpoint(filepath=checkpoint_path, save_freq=5)

mlflow_callback = MLflowCallback(val_data=(val_images, val_masks), num_images=10)

with mlflow.start_run():
    mlflow.keras.autolog()

    model.fit(
        train_generator,
        steps_per_epoch=len(train_images) // batch_size,
        epochs=20,
        validation_data=val_generator,
        validation_steps=len(val_images) // batch_size,
        callbacks=[checkpoint_callback, mlflow_callback]
    )

    mlflow.log_artifacts("save")