from datasets import load_dataset
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
from PIL import Image

dataset = load_dataset('Jackestar/RLCS_circuit_IEC_diagrams', download_mode="force_redownload")

validation_proportion = 0.15

# Apply train_test_split to the 'train' split within the DatasetDict
train_test_split = dataset['train'].train_test_split(test_size=validation_proportion)

# Update the DatasetDict with the split datasets
dataset['train'] = train_test_split['train']
dataset['validation'] = train_test_split['test']

print("Train Images:      " , dataset['train'].num_rows)
print("Validation Images: " , dataset['validation'].num_rows)
class_names = dataset['train'].features['label'].names

def preprocess_data(example):
    # Convertir la imagen de PIL a escala de grises y redimensionar
    image = example['image']
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize(resize_shape, Image.LANCZOS)  # Usar LANCZOS en lugar de ANTIALIAS

    # Convertir la imagen PIL a un tensor
    image = tf.convert_to_tensor(np.array(image))
    image = tf.expand_dims(image, axis=-1)  # Añadir una dimensión extra si es necesario
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalizar los valores de los píxeles
    image = tf.image.grayscale_to_rgb(image)  # Convertir a RGB para que coincida con el modelo preentrenado

    labels = example['label']  # Obtener las etiquetas
    labels = tf.one_hot(labels, depth=4)  # One-hot encoding de las etiquetas

    # Asegúrate de devolver un tensor para las bounding boxes si están presentes
    bboxes = example.get('bboxes', tf.zeros([0, 4], dtype=tf.float32))

    return image, labels  # Devolver imagen y etiquetas como una tupla

    # return {'image':image, 'labels':labels}  # Devolver imagen y etiquetas como una tupla

# Redimensionado
resize_shape = (224, 224)
shape=(224, 224, 3)

# Convertir el dataset de Hugging Face a tf.data.Dataset
train_data = tf.data.Dataset.from_generator(
    lambda: (preprocess_data(x) for x in dataset['train']),
    output_signature=(
        tf.TensorSpec(shape, dtype=tf.float32),
        tf.TensorSpec(shape=(4,), dtype=tf.float32)
    )
)

val_data = tf.data.Dataset.from_generator(
    lambda: (preprocess_data(x) for x in dataset['validation']),
    output_signature=(
        tf.TensorSpec(shape, dtype=tf.float32),
        tf.TensorSpec(shape=(4,), dtype=tf.float32)
    )
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32

train_data = train_data.shuffle(1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

# Descargar el modelo preentrenado desde TensorFlow Hub
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=shape, trainable=False)

# Definir el modelo usando transfer learning
model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(256, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_data, epochs=10, validation_data=val_data)

# Evaluar el modelo
loss, accuracy = model.evaluate(val_data)
print("Validation Loss: ", loss)
print("Validation Accuracy: ", accuracy)


model_name = "rlcmodel.hs"
print("Model saved as: ${model_name}")
model.save(model_name)