import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from xai.grad_cam import generate_grad_cam

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATA_DIR = 'dataset'

# Data loaders
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Functional API CNN Model
inputs = Input(shape=IMG_SIZE + (3,))
x = Conv2D(32, (3, 3), activation='relu', name='conv2d_0')(inputs)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu', name='conv2d_1')(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save
os.makedirs('model', exist_ok=True)
model.save('model/medical_model.h5')

# Evaluate
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Explain a sample image using Grad-CAM
img_path = test_data.filepaths[0]
generate_grad_cam(model, img_path, 'conv2d_1')  # You can change layer name if needed