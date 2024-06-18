import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras

from src.image_segmentation_model import ImageSegmentationModel
from callback import MLflowCallback  # Импортируйте ваш кастомный колбэк

# Настройка переменных окружения для аутентификации
os.environ['MLFLOW_TRACKING_USERNAME'] = 'mlflow'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '5u#K)R!@|<o==TXP<SlW'

# Настройка MLflow
mlflow.set_tracking_uri("https://mlflow.immoviewer.com")
mlflow.set_experiment("vladimir_semantic")

# Проверка доступа к серверу MLflow
try:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("vladimir_semantic")
    if experiment:
        print(f"Experiment '{experiment.name}' exists with ID {experiment.experiment_id}")
    else:
        print(f"Experiment 'vladimir_semantic' does not exist. Creating new experiment.")
        mlflow.create_experiment("vladimir_semantic")
except Exception as e:
    print(f"Failed to connect to MLflow server: {e}")
    sys.exit(1)

# Создание и компиляция модели
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

# Создание генераторов данных
batch_size = 16
train_generator = image_segmentation_model.data_generator(train_images, train_masks, batch_size)
val_generator = image_segmentation_model.data_generator(val_images, val_masks, batch_size)

# Установка колбэка
mlflow_callback = MLflowCallback(val_data=(val_images, val_masks), num_images=5)

# Обучение модели
with mlflow.start_run():
    mlflow.keras.autolog()

    model.fit(
        train_generator,
        steps_per_epoch=len(train_images) // batch_size,
        epochs=20,
        validation_data=val_generator,
        validation_steps=len(val_images) // batch_size,
        callbacks=[mlflow_callback]
    )
