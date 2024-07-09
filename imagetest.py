import sys
if len(sys.argv)<7:
    print("Faltan argumentos")
    print("python imagetest.py [classif_model_path] [ssd_model_path] [classname ...] [imagepath]")
    sys.exit(1)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import tensorflow_hub as hub
from ipywidgets import interact, IntSlider, FloatSlider
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# ['Capacitores', 'Fuentes', 'Inductores', 'Resistencia']
# 'Capacitores' 'Fuentes' 'Inductores' 'Resistencia'
class_names = sys.argv[3:7]

# Cargar tu modelo entrenado
# model_path = 'final_modelh2.h5'
model_path = sys.argv[1]
print(model_path)
# import keras
# model = keras.models.load_model(model_path)

# model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# ssd_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
ssd_path = sys.argv[2]
ssd_model = tf.saved_model.load(ssd_path)
classification_model = model

def detect_lines(image, threshold, min_line_length, max_line_gap, exclusion_zones, exclusion_padding):
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar desenfoque
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detección de bordes
    edges = cv2.Canny(blurred, 50, 150)

    # Enmascarar las zonas de exclusión
    mask = np.ones_like(edges) * 255
    for (startX, startY, endX, endY) in exclusion_zones:
      (startX, startY, endX, endY) = (startX+exclusion_padding, startY+exclusion_padding, endX-exclusion_padding, endY-exclusion_padding)
      mask[startY:endY, startX:endX] = 0
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Detección de líneas usando la Transformada de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

# Dibujar lineas sobre la imagen
def draw_lines(image, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

def merge_lines(lines, merge_threshold=10):
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        merged = False
        for other_line in merged_lines:
            ox1, oy1, ox2, oy2 = other_line[0]
            if (abs(x1 - ox1) < merge_threshold and abs(y1 - oy1) < merge_threshold) or (abs(x2 - ox2) < merge_threshold and abs(y2 - oy2) < merge_threshold):
                merged_line = [min(x1, ox1), min(y1, oy1), max(x2, ox2), max(y2, oy2)]
                other_line[0] = merged_line
                merged = True
                break
        if not merged:
            merged_lines.append(line)
    return merged_lines

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=len(boxes), iou_threshold=iou_threshold)
    return indices.numpy()

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    # Convertir la imagen a un tipo de datos apropiado
    image = np.int16(image)
    # Ajustar brillo
    image = image * (contrast / 127 + 1) - contrast + brightness
    # Volver a convertir a uint8
    image = np.clip(image, 0, 255)
    return np.uint8(image)

def detect_and_classify(image_path, ssd_model, classification_model, classes,
                        min_confidence=0.5, size_threshold=2.0, threshold=100,
                        min_line_length=50, max_line_gap=10, brightness=0, contrast=0,merge_threshold=10,exclusion_padding=0,iou_threshold=0.5):
    image_size = (224, 224)  # Define un tamaño de entrada compatible con el modelo de clasificación

    # Cargar la imagen
    image = cv2.imread(image_path)

    # Ajustar brillo y contraste
    image = adjust_brightness_contrast(image, brightness, contrast)

    orig_image = image.copy()
    (h, w) = image.shape[:2]

    # Convertir la imagen a formato adecuado para el modelo de detección
    # image_resized = cv2.resize(image, (600, 600))
    image_resized = image
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_resized = np.expand_dims(image_resized, axis=0)

    # Detección de objetos
    detector_output = ssd_model(image_resized)
    detection_boxes = detector_output['detection_boxes'][0].numpy()
    detection_classes = detector_output['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detector_output['detection_scores'][0].numpy()

    detected_objects = []
    for i in range(len(detection_scores)):
        if detection_scores[i] > min_confidence:  # Umbral de confianza
            box = detection_boxes[i] * np.array([h, w, h, w])
            (startY, startX, endY, endX) = box.astype("int")

            width = endX - startX
            height = endY - startY
            area = width * height

            detected_objects.append({
                'class': detection_classes[i],
                'confidence': detection_scores[i],
                'box': (startX, startY, endX, endY),
                'area': area
            })
    # print(detection_scores)
    # Normalización del tamaño de detección
    if detected_objects:
        areas = [obj['area'] for obj in detected_objects]
        mean_area = np.mean(areas)
        filtered_objects = [obj for obj in detected_objects if obj['area'] <= size_threshold * mean_area]
    else:
        filtered_objects = []

    # Supresión de no-máximos para eliminar superposiciones
    if filtered_objects:
        boxes = np.array([obj['box'] for obj in filtered_objects])
        scores = np.array([obj['confidence'] for obj in filtered_objects])
        indices = non_max_suppression(boxes, scores, iou_threshold)
        final_objects = [filtered_objects[i] for i in indices]
    else:
        final_objects = []

    # Detectar líneas excluyendo las regiones de los objetos detectados
    exclusion_zones = [obj['box'] for obj in final_objects]
    lines = detect_lines(orig_image, threshold, min_line_length, max_line_gap, exclusion_zones,exclusion_padding)

    # Unir las líneas que se toquen entre sí
    if lines is not None:
        lines = merge_lines(lines,merge_threshold)

    # Dibujar líneas
    if lines is not None:
        draw_lines(orig_image, lines)

    # Clasificar objetos y verificar conexiones por líneas
    connected_objects = []
    for obj in final_objects:
        (startX, startY, endX, endY) = obj['box']

        # Extraer la región detectada
        roi = orig_image[startY:endY, startX:endX]
        if roi.size == 0:
            continue
        roi_resized = cv2.resize(roi, image_size)  # Redimensionar la región de interés
        roi_resized = roi_resized.astype("float32") / 255.0
        roi_resized = np.expand_dims(roi_resized, axis=0)

        # Clasificación de la región
        preds = classification_model.predict(roi_resized)
        predicted_class = classes[np.argmax(preds)]
        confidence_score = np.max(preds)

        # Dibujar el cuadro en la imagen original
        label = f"{predicted_class}: {confidence_score * 100:.2f}%"
        cv2.rectangle(orig_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(orig_image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        obj['predicted_class'] = predicted_class
        obj['confidence_score'] = confidence_score
        connected_objects.append(obj)

    # Verificar si los objetos están conectados por líneas
    connections = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                connected_pairs = []
                for obj in connected_objects:
                    (startX, startY, endX, endY) = obj['box']
                    if (startX <= x1 <= endX and startY <= y1 <= endY) or (startX <= x2 <= endX and startY <= y2 <= endY):
                        connected_pairs.append(obj)
                if len(connected_pairs) > 1:
                    for i in range(len(connected_pairs) - 1):
                        connections.append((connected_pairs[i], connected_pairs[i + 1]))

    # Agrupar objetos conectados
    connected_groups = []
    for connection in connections:
        found_group = False
        for group in connected_groups:
            if connection[0] in group or connection[1] in group:
                group.update(connection)
                found_group = True
                break
        if not found_group:
            connected_groups.append(set(connection))

    # Mostrar la imagen con las detecciones
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Devolver resultados
    for group_idx, group in enumerate(connected_groups, 1):
        for obj in group:
            connections_text = f"Grupo {group_idx}"
            print(f"Objeto {connected_objects.index(obj) + 1}: Clase detectada: {obj['predicted_class']}, Confianza: {obj['confidence_score'] * 100:.2f}%, Ubicación: {obj['box']} {connections_text}")

    for i, obj in enumerate(connected_objects):
        connections_text = ""
        for connection in connections:
            if connection[0] == obj:
                connected_obj = connection[1]
                connections_text += f" Conectado a Objeto {connected_objects.index(connected_obj) + 1}"
            elif connection[1] == obj:
                connected_obj = connection[0]
                connections_text += f" Conectado a Objeto {connected_objects.index(connected_obj) + 1}"
        print(f"Objeto {i + 1}: Clase detectada: {obj['predicted_class']}, Confianza: {obj['confidence_score'] * 100:.2f}%, Ubicación: {obj['box']} {connections_text}")

# Definir la función interactiva
def interactive_detection(image_path, min_confidence, size_threshold, threshold, min_line_length, max_line_gap, brightness, contrast,merge_threshold,exclusion_padding,iou_threshold):
    detect_and_classify(image_path, ssd_model, classification_model, class_names,
                        min_confidence, size_threshold, threshold,
                        min_line_length, max_line_gap, brightness, contrast,merge_threshold,exclusion_padding,iou_threshold)

image_path = sys.argv[8]
image_size = (224,224)
interact(interactive_detection,
         image_path=image_path,
         min_confidence=FloatSlider(value=0.15, min=0.0, max=1.0, step=0.01, description='Min Conf.'),
         size_threshold=FloatSlider(value=0.7, min=0.1, max=2.0, step=0.1, description='Size Threshold'),
         threshold=IntSlider(value=30, min=1, max=150, step=1, description='Threshold'),
         min_line_length=IntSlider(value=20, min=1, max=100, step=1, description='Min Line Length'),
         max_line_gap=IntSlider(value=25, min=1, max=50, step=1, description='Max Line Gap'),
         brightness=IntSlider(value=0, min=1, max=255, step=5, description='brigthness'),
         contrast=IntSlider(value=0, min=1, max=255, step=5, description='contrast'),
         merge_threshold=IntSlider(value=10, min=0, max=255, step=5, description='mergethreshold'),
         exclusion_padding=IntSlider(value=0, min=0, max=255, step=5, description='exclusion padding'),
         iou_threshold=FloatSlider(value=0.5, min=0, max=5, step=0.5, description='iou threshold'))